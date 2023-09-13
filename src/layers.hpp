#ifndef LAYERS_HPP
#define LAYERS_HPP

#define start_ auto start = std::chrono::high_resolution_clock::now();
#define stop_                                             \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::cout << "Elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count() << "ms" << std::endl;

#define Tensor1D std::vector<T>
#define Tensor3D std::vector<std::vector<std::vector<T>>>

#include <future>
#include <string>
#include <vector>

#include "AFloat.hpp"
#include "onnx.proto3.pb.h"
#include "threadpool.hpp"
#include <fmt/core.h>

ThreadPool pool( 64 );

namespace torch {
template <typename T>
class Conv2d {
public:
    Conv2d() {}
    Conv2d( int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation = 1, int groups = 1, bool bias = true ) {
        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->groups = groups;
        this->bias = bias;
    }
    void load( const onnx::TensorProto &weight, const onnx::TensorProto &bias__ ) {
        in_channels = weight.dims( 1 );
        out_channels = weight.dims( 0 );
        kernel_size = weight.dims( 2 );
        auto weight_start = weight.float_data_size() ? weight.float_data().begin() : reinterpret_cast<const float *>( weight.raw_data().data() );
        auto weight_end = weight.float_data_size() ? weight.float_data().end() : reinterpret_cast<const float *>( weight.raw_data().data() ) + weight.raw_data().size() / sizeof( float );
        this->weight = Tensor1D( std::distance( weight_start, weight_end ) );
        std::transform( weight_start, weight_end, this->weight.begin(), []( float f ) { return T( f ); } );
        auto bias_start = bias__.float_data_size() ? bias__.float_data().begin() : reinterpret_cast<const float *>( bias__.raw_data().data() );
        auto bias_end = bias__.float_data_size() ? bias__.float_data().end() : reinterpret_cast<const float *>( bias__.raw_data().data() ) + bias__.raw_data().size() / sizeof( float );
        this->bias_ = Tensor1D( std::distance( bias_start, bias_end ) );
        std::transform( bias_start, bias_end, this->bias_.begin(), []( float f ) { return T( f ); } );
    }

    Tensor3D forward( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        int input_size = input[ 0 ].size();
        int output_size = static_cast<int>( std::floor( 1 + ( input_size + 2 * padding - dilation * ( kernel_size - 1 ) - 1 ) / stride ) );

        Tensor3D output;
        output.resize( out_channels );
        for ( auto &i : output ) {
            i.resize( output_size );
            for ( auto &j : i ) {
                j.resize( output_size );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }
        std::vector<std::future<void>> futures;
        futures.reserve( out_channels );
        // Convolution operation
        for ( int oc = 0; oc < out_channels; oc++ ) {
            futures.emplace_back( pool.enqueue( [ oc, &input, &output, this, output_size, input_size ]() {
                for ( int ic = 0; ic < in_channels; ic++ ) {
                    for ( int i = 0; i < output_size; i++ ) {
                        for ( int j = 0; j < output_size; j++ ) {
                            for ( int m = 0; m < kernel_size; m++ ) {
                                for ( int n = 0; n < kernel_size; n++ ) {
                                    int x = i * stride - padding + m * dilation;
                                    int y = j * stride - padding + n * dilation;

                                    if ( x >= 0 && x < input_size && y >= 0 && y < input_size ) {
                                        output[ oc ][ i ][ j ] += input[ ic ][ x ][ y ] * weight[ oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + m * kernel_size + n ];
                                    }
                                }
                            }
                        }
                    }
                }

                if ( bias ) {
                    for ( int i = 0; i < output_size; i++ ) {
                        for ( int j = 0; j < output_size; j++ ) {
                            output[ oc ][ i ][ j ] += bias_[ oc ];
                        }
                    }
                }
            } ) );
        }
        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

#ifdef DEBUG
        stop_;
#endif

        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward( input );
    }

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int groups;
    bool bias;

    // private:
    Tensor1D weight;
    Tensor1D bias_;
};

template <typename T>
class ReLU {
public:
    ReLU() {}

    Tensor3D forward_multithread( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( input[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( input[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );

        for ( int i = 0; i < input.size(); i++ ) {
            futures.emplace_back( pool.enqueue( [ &input, &output ]( int i ) {
                for ( int j = 0; j < input[ 0 ].size(); j++ ) {
                    for ( int k = 0; k < input[ 0 ][ 0 ].size(); k++ ) {
                        output[ i ][ j ][ k ] = input[ i ][ j ][ k ] > T( 0.0f ) ? input[ i ][ j ][ k ] : T( 0.0f );
                    }
                }
                return;
            },
                                                i ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

#ifdef DEBUG
        stop_;
#endif
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward_multithread( input );
    }
};

template <typename T>
class LeakyReLU {
public:
    LeakyReLU( T slope = T( 0.2f ) ) {
        this->slope = slope;
    }

    Tensor3D forward_multithread( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( input[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( input[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );
        for ( int i = 0; i < input.size(); i++ ) {
            futures.emplace_back( pool.enqueue( [ &input, this ]( int i ) {
                for ( int j = 0; j < input[ 0 ].size(); j++ ) {
                    for ( int k = 0; k < input[ 0 ][ 0 ].size(); k++ ) {
                        input[ i ][ j ][ k ] = input[ i ][ j ][ k ] > T( 0.0f ) ? input[ i ][ j ][ k ] : slope * input[ i ][ j ][ k ];
                    }
                }
                return;
            },
                                                i ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

#ifdef DEBUG
        stop_;
#endif
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward_multithread( input );
    }

private:
    T slope;
};

template <typename T>
class Tanh {
public:
    Tanh() {}

    Tensor3D forward_multithread( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( input[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( input[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );

        for ( int i = 0; i < input.size(); i++ ) {
            futures.emplace_back( pool.enqueue( [ &output, input, this, i ] {
                for ( int j = 0; j < input[ 0 ].size(); j++ ) {
                    for ( int k = 0; k < input[ 0 ][ 0 ].size(); k++ ) {
                        output[ i ][ j ][ k ] = tanh( input[ i ][ j ][ k ] );
                    }
                }
                return;
            } ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

#ifdef DEBUG
        stop_;
#endif
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward_multithread( input );
    }
};

template <typename T>
class ConvTranspose2d {
public:
    ConvTranspose2d( int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, int output_padding = 0, int dilation = 1, int groups = 1, bool bias = true ) {
        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->output_padding = output_padding;
        this->dilation = dilation;
        this->groups = groups;
        this->bias = bias;
    }
    void load( Tensor1D &weight, Tensor1D &bias__ ) {
        this->weight = std::move( weight );
        this->bias_ = std::move( bias__ );
    }
    void load( T *weight, int size ) {
        this->weight = Tensor1D( weight, weight + size );
    }
    void load( const onnx::TensorProto &weight, const onnx::TensorProto &bias__ ) {
        in_channels = weight.dims( 0 );
        out_channels = weight.dims( 1 );
        kernel_size = weight.dims( 2 );
        auto weight_start = weight.float_data_size() ? weight.float_data().begin() : reinterpret_cast<const float *>( weight.raw_data().data() );
        auto weight_end = weight.float_data_size() ? weight.float_data().end() : reinterpret_cast<const float *>( weight.raw_data().data() ) + weight.raw_data().size() / sizeof( float );
        this->weight = Tensor1D( std::distance( weight_start, weight_end ) );
        std::transform( weight_start, weight_end, this->weight.begin(), []( float f ) { return T( f ); } );
        auto bias_start = bias__.float_data_size() ? bias__.float_data().begin() : reinterpret_cast<const float *>( bias__.raw_data().data() );
        auto bias_end = bias__.float_data_size() ? bias__.float_data().end() : reinterpret_cast<const float *>( bias__.raw_data().data() ) + bias__.raw_data().size() / sizeof( float );
        this->bias_ = Tensor1D( std::distance( bias_start, bias_end ) );
        std::transform( bias_start, bias_end, this->bias_.begin(), []( float f ) { return T( f ); } );
    }

    Tensor3D forward( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        int output_size = ( input[ 0 ].size() - 1 ) * stride - 2 * padding + dilation * ( kernel_size - 1 ) + output_padding + 1;
        Tensor3D output;
        output.resize( out_channels );
        for ( auto &i : output ) {
            i.resize( output_size );
            for ( auto &j : i ) {
                j.resize( output_size );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( out_channels );

        // Deconvolution operation
        for ( int oc = 0; oc < out_channels; oc++ ) {
            futures.emplace_back( pool.enqueue( [ oc, &input, &output, this, output_size ]() {
                for ( int ic = 0; ic < in_channels; ic++ ) {
                    for ( int i = 0; i < input[ ic ].size(); i++ ) {
                        for ( int j = 0; j < input[ ic ][ i ].size(); j++ ) {
                            for ( int m = 0; m < kernel_size; m++ ) {
                                for ( int n = 0; n < kernel_size; n++ ) {
                                    int x = i * stride - padding + m * dilation;
                                    int y = j * stride - padding + n * dilation;
                                    if ( x >= 0 && x < output_size && y >= 0 && y < output_size ) {
                                        output[ oc ][ x ][ y ] += input[ ic ][ i ][ j ] * weight[ ic * out_channels * kernel_size * kernel_size + oc * kernel_size * kernel_size + m * kernel_size + n ];
                                    }
                                }
                            }
                        }
                    }
                }

                if ( bias ) {
                    for ( int i = 0; i < output_size; i++ ) {
                        for ( int j = 0; j < output_size; j++ ) {
                            output[ oc ][ i ][ j ] += bias_[ oc ];
                        }
                    }
                }
            } ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.wait();
        }

#ifdef DEBUG
        stop_;
#endif

        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward( input );
    }

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int output_padding;
    int dilation;
    int groups;
    bool bias;

private:
    Tensor1D weight;
    Tensor1D bias_;
};

template <typename T>
class InstanceNorm2d {
public:
    InstanceNorm2d( int num_features, T eps = T( 1e-05 ), T momentum = T( 0.1f ), bool affine = false ) {
        this->num_features = num_features;
        this->eps = eps;
        this->momentum = momentum;
        this->affine = affine;
    }
    void load( Tensor1D &gamma, Tensor1D &beta ) {
        this->gamma = std::move( gamma );
        this->beta = std::move( beta );
        this->mean = std::move( mean );
        this->var = std::move( var );
    }
    void load( const onnx::TensorProto &gamma, const onnx::TensorProto &beta ) {
        num_features = beta.dims( 0 );
        auto beta_start = beta.float_data_size() ? beta.float_data().begin() : reinterpret_cast<const float *>( beta.raw_data().data() );
        auto beta_end = beta.float_data_size() ? beta.float_data().end() : reinterpret_cast<const float *>( beta.raw_data().data() ) + beta.raw_data().size() / sizeof( float );
        this->beta = Tensor1D( beta_start, beta_end );
        std::transform( beta_start, beta_end, this->beta.begin(), []( float f ) { return T( f ); } );
        auto gamma_start = gamma.float_data_size() ? gamma.float_data().begin() : reinterpret_cast<const float *>( gamma.raw_data().data() );
        auto gamma_end = gamma.float_data_size() ? gamma.float_data().end() : reinterpret_cast<const float *>( gamma.raw_data().data() ) + gamma.raw_data().size() / sizeof( float );
        this->gamma = Tensor1D( gamma_start, gamma_end );
        std::transform( gamma_start, gamma_end, this->gamma.begin(), []( float f ) { return T( f ); } );
    }
    Tensor3D forward_multithread( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( input[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( input[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( num_features );

        T total_size = T( static_cast<float>( input[ 0 ].size() * input[ 0 ][ 0 ].size() ) );
        for ( int i = 0; i < num_features; i++ ) {
            futures.emplace_back( pool.enqueue( [ &input, this, total_size ]( int j ) {
                T mean = T( 0.0f );
                T square_mean = T( 0.0f );
                for ( int k = 0; k < input[ 0 ].size(); k++ ) {
                    for ( int l = 0; l < input[ 0 ][ 0 ].size(); l++ ) {
                        mean += input[ j ][ k ][ l ] / total_size;
                        square_mean += input[ j ][ k ][ l ] * input[ j ][ k ][ l ] / total_size;
                    }
                }
                T var = square_mean - mean * mean;
                for ( int k = 0; k < input[ 0 ].size(); k++ ) {
                    for ( int l = 0; l < input[ 0 ][ 0 ].size(); l++ ) {
                        input[ j ][ k ][ l ] = ( input[ j ][ k ][ l ] - mean ) / sqrt( var + eps );
                        input[ j ][ k ][ l ] = input[ j ][ k ][ l ] * gamma[ j ] + beta[ j ];
                    }
                }
                return;
            },
                                                i ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

#ifdef DEBUG
        stop_;
#endif
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward_multithread( input );
    }

private:
    int num_features;
    T eps;
    T momentum;
    bool affine;

    Tensor1D gamma;
    Tensor1D beta;
    Tensor1D mean;
    Tensor1D var;
};

template <typename T>
class BatchNorm2d {
public:
    BatchNorm2d( int num_features, T eps = T( 1e-05 ), T momentum = T( 0.1f ), bool affine = false ) {
        this->num_features = num_features;
        this->eps = eps;
        this->momentum = momentum;
        this->affine = affine;
    }
    void load( Tensor1D &mean, Tensor1D &var ) {
        this->mean = std::move( mean );
        this->var = std::move( var );
    }
    void load( Tensor1D &gamma, Tensor1D &beta, Tensor1D &mean, Tensor1D &var ) {
        this->gamma = std::move( gamma );
        this->beta = std::move( beta );
        this->mean = std::move( mean );
        this->var = std::move( var );
    }
    void load( const onnx::TensorProto &mean, const onnx::TensorProto &var, const onnx::TensorProto &gamma, const onnx::TensorProto &beta ) {
        num_features = mean.dims( 0 );
        auto mean_start = mean.float_data_size() ? mean.float_data().begin() : reinterpret_cast<const float *>( mean.raw_data().data() );
        auto mean_end = mean.float_data_size() ? mean.float_data().end() : reinterpret_cast<const float *>( mean.raw_data().data() ) + mean.raw_data().size() / sizeof( float );
        this->mean = Tensor1D( std::distance( mean_start, mean_end ) );
        std::transform( mean_start, mean_end, this->mean.begin(), []( float f ) { return T( f ); } );
        auto var_start = var.float_data_size() ? var.float_data().begin() : reinterpret_cast<const float *>( var.raw_data().data() );
        auto var_end = var.float_data_size() ? var.float_data().end() : reinterpret_cast<const float *>( var.raw_data().data() ) + var.raw_data().size() / sizeof( float );
        this->var = Tensor1D( std::distance( var_start, var_end ) );
        std::transform( var_start, var_end, this->var.begin(), []( float f ) { return T( f ); } );
        auto beta_start = beta.float_data_size() ? beta.float_data().begin() : reinterpret_cast<const float *>( beta.raw_data().data() );
        auto beta_end = beta.float_data_size() ? beta.float_data().end() : reinterpret_cast<const float *>( beta.raw_data().data() ) + beta.raw_data().size() / sizeof( float );
        this->beta = Tensor1D( std::distance( beta_start, beta_end ) );
        std::transform( beta_start, beta_end, this->beta.begin(), []( float f ) { return T( f ); } );
        auto gamma_start = gamma.float_data_size() ? gamma.float_data().begin() : reinterpret_cast<const float *>( gamma.raw_data().data() );
        auto gamma_end = gamma.float_data_size() ? gamma.float_data().end() : reinterpret_cast<const float *>( gamma.raw_data().data() ) + gamma.raw_data().size() / sizeof( float );
        this->gamma = Tensor1D( std::distance( gamma_start, gamma_end ) );
        std::transform( gamma_start, gamma_end, this->gamma.begin(), []( float f ) { return T( f ); } );
    }
    Tensor3D forward_multithread( Tensor3D &input ) {
#ifdef DEBUG
        start_;
#endif

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( input[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( input[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );

        for ( int i = 0; i < num_features; i++ ) {
            futures.emplace_back( pool.enqueue( [ &output, &input, this ]( int j ) {
                for ( int k = 0; k < input[ 0 ].size(); k++ ) {
                    for ( int l = 0; l < input[ 0 ][ 0 ].size(); l++ ) {
                        output[ j ][ k ][ l ] = ( input[ j ][ k ][ l ] - mean[ j ] ) / sqrt( var[ j ] + eps );
                        output[ j ][ k ][ l ] = input[ j ][ k ][ l ] * gamma[ j ] + beta[ j ];
                    }
                }
                return;
            },
                                                i ) );
        }
#ifdef DEBUG
        stop_;
#endif
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward_multithread( input );
    }

private:
    int num_features;
    T eps;
    T momentum;
    bool affine;

    Tensor1D gamma;
    Tensor1D beta;
    Tensor1D mean;
    Tensor1D var;
};

template <typename T>
class MaxPool2d {
public:
    MaxPool2d( int kernel_size, int stride, int padding = 0, int dilation = 1 ) {
        this->kernel_size = kernel_size;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
    }

    Tensor3D forward( Tensor3D &input ) {
        int input_size = input[ 0 ].size();
        int output_size = static_cast<int>( std::floor( 1 + ( input_size + 2 * padding - dilation * ( kernel_size - 1 ) - 1 ) / stride ) );

        Tensor3D output;
        output.resize( input.size() );
        for ( auto &i : output ) {
            i.resize( output_size );
            for ( auto &j : i ) {
                j.resize( output_size );
                for ( auto &k : j ) {
                    k = 0.0f;
                }
            }
        }
        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );
        for ( int ic = 0; ic < input.size(); ic++ ) {
            futures.emplace_back( pool.enqueue( [ &input, &output, this, output_size, input_size, ic ]() {
                for ( int i = 0; i < output_size; i++ ) {
                    for ( int j = 0; j < output_size; j++ ) {
                        T max = 0.0f;
                        bool initialized = false;
                        for ( int m = 0; m < kernel_size; m++ ) {
                            for ( int n = 0; n < kernel_size; n++ ) {
                                int x = i * stride - padding + m * dilation;
                                int y = j * stride - padding + n * dilation;

                                if ( x >= 0 && x < input_size && y >= 0 && y < input_size ) {
                                    if ( !initialized ) {
                                        max = input[ ic ][ x ][ y ];
                                        initialized = true;
                                    } else
                                        max = std::max<>( max, input[ ic ][ x ][ y ] );
                                }
                            }
                        }
                        output[ ic ][ i ][ j ] = max;
                    }
                }
            } ) );
        }
        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward( input );
    }

private:
    int kernel_size;
    int stride;
    int padding;
    int dilation;
};

template <typename T>
class Add {
public:
    Add() {}

    Tensor3D forward_multithread( Tensor3D &a, Tensor3D &b ) {
        Tensor3D output;
        output.resize( a.size() );
        for ( auto &i : output ) {
            i.resize( a[ 0 ].size() );
            for ( auto &j : i ) {
                j.resize( a[ 0 ][ 0 ].size() );
                for ( auto &k : j ) {
                    k = T( 0.0 );
                }
            }
        }
        std::vector<std::future<void>> futures;
        futures.reserve( a.size() );
        for ( int i = 0; i < a.size(); i++ ) {
            futures.emplace_back( pool.enqueue( [ &a, &b, &output ]( int i ) {
                for ( int j = 0; j < a[ 0 ].size(); j++ ) {
                    for ( int k = 0; k < a[ 0 ][ 0 ].size(); k++ ) {
                        output[ i ][ j ][ k ] = a[ i ][ j ][ k ] + b[ i ][ j ][ k ];
                    }
                }
                return;
            },
                                                i ) );
        }
        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }
        return output;
    }
    Tensor3D operator()( Tensor3D &a, Tensor3D &b ) {
        return forward_multithread( a, b );
    }
};

template <typename T>
class Identity {
public:
    Identity() {}

    Tensor3D forward( Tensor3D &input ) {
        Tensor3D output( input );
        return output;
    }
    Tensor3D operator()( Tensor3D &input ) {
        return forward( input );
    }
};

template <typename T>
class Gemm {
public:
    Gemm( int in_dim, int out_dim ) {
        this->in_dim = in_dim;
        this->out_dim = out_dim;
    }

    void load( Tensor1D &weight, Tensor1D &bias__ ) {
        this->weight = std::move( weight );
        this->bias_ = std::move( bias__ );
    }

    void load( T *weight, int size ) {
        this->weight = Tensor1D( weight, weight + size );
    }

    void load( const onnx::TensorProto &weight, const onnx::TensorProto &bias__ ) {
        in_dim = weight.dims( 1 );
        out_dim = weight.dims( 0 );
        auto weight_start = weight.float_data_size() ? weight.float_data().begin() : reinterpret_cast<const float *>( weight.raw_data().data() );
        auto weight_end = weight.float_data_size() ? weight.float_data().end() : reinterpret_cast<const float *>( weight.raw_data().data() ) + weight.raw_data().size() / sizeof( float );
        this->weight = Tensor1D( std::distance( weight_start, weight_end ) );
        std::transform( weight_start, weight_end, this->weight.begin(), []( float f ) { return T( f ); } );
        auto bias_start = bias__.float_data_size() ? bias__.float_data().begin() : reinterpret_cast<const float *>( bias__.raw_data().data() );
        auto bias_end = bias__.float_data_size() ? bias__.float_data().end() : reinterpret_cast<const float *>( bias__.raw_data().data() ) + bias__.raw_data().size() / sizeof( float );
        this->bias_ = Tensor1D( std::distance( bias_start, bias_end ) );
        std::transform( bias_start, bias_end, this->bias_.begin(), []( float f ) { return T( f ); } );
    }

    Tensor1D forward( Tensor1D &input ) {
        Tensor1D output;
        output.resize( out_dim );

        std::vector<std::future<void>> futures;
        futures.reserve( out_dim );

        for ( int i = 0; i < out_dim; i++ ) {
            futures.emplace_back( pool.enqueue( [ &input, &output, this ]( int i ) {
                output[ i ] = bias_[ i ];
                for ( int j = 0; j < in_dim; j++ ) {
                    output[ i ] += input[ j ] * weight[ i * in_dim + j ];
                }
                return;
            },
                                                i ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }

        return output;
    }
    Tensor1D operator()( Tensor1D &input ) {
        return forward( input );
    }

private:
    int in_dim;
    int out_dim;

    Tensor1D weight;
    Tensor1D bias_;
};

template <typename T>
class Flatten {
public:
    Flatten() {}

    Tensor1D forward( Tensor3D &input ) {
        Tensor1D output;
        output.resize( input.size() * input[ 0 ].size() * input[ 0 ][ 0 ].size() );
        int index = 0;
        for ( int i = 0; i < input.size(); i++ ) {
            for ( int j = 0; j < input[ 0 ].size(); j++ ) {
                for ( int k = 0; k < input[ 0 ][ 0 ].size(); k++ ) {
                    output[ index++ ] = input[ i ][ j ][ k ];
                }
            }
        }
        return output;
    }

    Tensor1D operator()( Tensor3D &input ) {
        return forward( input );
    }
};

template <typename T>
class GAP {
public:
    GAP() {}

    Tensor3D forward( Tensor3D &input ) {
        Tensor3D output;
        output.resize( input.size() );

        for ( auto &i : output ) {
            i.resize( 1 );
            i[ 0 ].resize( 1 );
        }

        std::vector<std::future<void>> futures;
        futures.reserve( input.size() );

        T total_size = T( static_cast<float>( input[ 0 ].size() * input[ 0 ][ 0 ].size() ) );
        for ( int i = 0; i < input.size(); i++ ) {
            futures.emplace_back( pool.enqueue( [ &input, &output, total_size, this ]( int i ) {
                output[ i ][ 0 ][ 0 ] = T( 0.0f );
                for ( int j = 0; j < input[ 0 ].size(); j++ ) {
                    for ( int k = 0; k < input[ 0 ][ 0 ].size(); k++ ) {
                        output[ i ][ 0 ][ 0 ] += input[ i ][ j ][ k ];
                    }
                }
                output[ i ][ 0 ][ 0 ] /= total_size;
                return;
            },
                                                i ) );
        }

        // Wait for all futures to complete
        for ( auto &future : futures ) {
            future.get();
        }
        return output;
    }

    Tensor3D operator()( Tensor3D &input ) {
        return forward( input );
    }

}; // GAP
} // namespace torch

template <typename T>
Tensor3D operator+( Tensor3D &a, Tensor3D &b ) {
    std::cout << "operator+ called" << std::endl;
    Tensor3D output;
    output.resize( a.size() );
    for ( auto &i : output ) {
        i.resize( a[ 0 ].size() );
        for ( auto &j : i ) {
            j.resize( a[ 0 ][ 0 ].size() );
            for ( auto &k : j ) {
                k = T( 0.0 );
            }
        }
    }
    std::vector<std::future<void>> futures;
    futures.reserve( a.size() );
    for ( int i = 0; i < a.size(); i++ ) {
        futures.emplace_back( pool.enqueue( [ &a, &b, &output ]( int i ) {
            for ( int j = 0; j < a[ 0 ].size(); j++ ) {
                for ( int k = 0; k < a[ 0 ][ 0 ].size(); k++ ) {
                    output[ i ][ j ][ k ] = a[ i ][ j ][ k ] + b[ i ][ j ][ k ];
                }
            }
            return;
        },
                                            i ) );
    }
    // Wait for all futures to complete
    for ( auto &future : futures ) {
        future.get();
    }
    return output;
}
#endif // LAYERS_HPP