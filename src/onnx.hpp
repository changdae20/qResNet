#include "AFloat.hpp"
#include "layers.hpp"
#include "onnx.proto3.pb.h"
#include "threadpool.hpp"
#include "util.hpp"
#include <bits/stdc++.h>

template <typename T>
using layer = std::variant<
    torch::Conv2d<T>,
    torch::BatchNorm2d<T>,
    torch::ConvTranspose2d<T>,
    torch::InstanceNorm2d<T>,
    torch::LeakyReLU<T>,
    torch::MaxPool2d<T>,
    torch::ReLU<T>,
    torch::Tanh<T>,
    torch::Add<T>,
    torch::Identity<T>,
    torch::GAP<T>,
    torch::Flatten<T>,
    torch::Gemm<T>>;

template <typename T>
class Onnx {
public:
    Onnx() = delete;
    Onnx( std::string filename ) {
        std::ifstream in( filename.c_str(), std::ios::ate | std::ios::binary );

        std::streamsize size = in.tellg();
        in.seekg( 0, std::ios::beg );

        std::vector<char> buffer( size );
        in.read( buffer.data(), size );

        model.ParseFromArray( buffer.data(), size );
        graph = model.graph();

        std::set<std::string> ready_inputs;
        std::map<std::string, const onnx::TensorProto> parameters;

        for ( int i = 0; i < graph.input_size(); ++i ) {
            auto input = graph.input( i );
            std::string name = input.name();
            ready_inputs.insert( name );
        }
        for ( int i = 0; i < graph.initializer_size(); ++i ) {
            std::string name = graph.initializer( i ).name();
#ifdef DEBUG
            std::cout << fmt::format( "initializer({}) : {}", i, name ) << std::endl;
#endif
            ready_inputs.insert( name );
            parameters.insert( { name, graph.initializer( i ) } );
        }
        for ( int i = 0; i < graph.node_size(); ++i ) {
#ifdef DEBUG
            std::cout << fmt::format( "({}) : {}", i, graph.node( i ).name() ) << std::endl;
#endif
            auto node = graph.node( i );
            std::string op_type = node.op_type();
            std::string name = node.name();
            if ( op_type == "Conv" ) {
                int in_channels, out_channels, dilations, group, kernel_shape, pads, strides;
                in_channels = out_channels = dilations = group = kernel_shape = pads = strides = 0;
                auto s = node.input( 1 );
                if ( auto it = std::find_if( graph.initializer().begin(), graph.initializer().end(), [ s ]( auto &el ) { return el.name() == s; } ); it != graph.initializer().end() ) {
                    in_channels = it->dims( 1 );
                    out_channels = it->dims( 0 );
                }
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "dilations" )
                        dilations = attribute.ints( 0 );
                    else if ( attribute.name() == "group" )
                        group = attribute.i();
                    else if ( attribute.name() == "kernel_shape" )
                        kernel_shape = attribute.ints( 0 );
                    else if ( attribute.name() == "pads" )
                        pads = attribute.ints( 0 );
                    else if ( attribute.name() == "strides" )
                        strides = attribute.ints( 0 );
                }
                layers.emplace_back( name, torch::Conv2d<T>( in_channels, out_channels, kernel_shape, strides, pads, dilations, group ) );
                std::get<0>( layers.back().second ).load( parameters[ node.input( 1 ) ], parameters[ node.input( 2 ) ] );
            }

            else if ( op_type == "BatchNormalization" ) {
                int in_channels = 0;
                auto s = node.input( 1 );
                if ( auto it = std::find_if( graph.initializer().begin(), graph.initializer().end(), [ s ]( auto &el ) { return el.name() == s; } ); it != graph.initializer().end() ) {
                    in_channels = it->dims( 0 );
                }
                T epsilon, momentum;
                epsilon = momentum = 0.0f;
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "epsilon" )
                        epsilon = attribute.f();
                    else if ( attribute.name() == "momentum" )
                        momentum = attribute.f();
                }
                layers.emplace_back( name, torch::BatchNorm2d<T>( in_channels, epsilon, momentum ) );
                std::get<1>( layers.back().second ).load( parameters[ node.input( 3 ) ], parameters[ node.input( 4 ) ], parameters[ node.input( 1 ) ], parameters[ node.input( 2 ) ] );
            } else if ( op_type == "ConvTranspose" ) {
                int in_channels, out_channels, dilations, group, kernel_shape, pads, strides;
                in_channels = out_channels = dilations = group = kernel_shape = pads = strides = 0;
                auto s = node.input( 1 );
                if ( auto it = std::find_if( graph.initializer().begin(), graph.initializer().end(), [ s ]( auto &el ) { return el.name() == s; } ); it != graph.initializer().end() ) {
                    in_channels = it->dims( 1 );
                    out_channels = it->dims( 0 );
                }
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "dilations" )
                        dilations = attribute.ints( 0 );
                    else if ( attribute.name() == "group" )
                        group = attribute.i();
                    else if ( attribute.name() == "kernel_shape" )
                        kernel_shape = attribute.ints( 0 );
                    else if ( attribute.name() == "pads" )
                        pads = attribute.ints( 0 );
                    else if ( attribute.name() == "strides" )
                        strides = attribute.ints( 0 );
                }
                layers.emplace_back( name, torch::ConvTranspose2d<T>( in_channels, out_channels, kernel_shape, strides, pads, dilations, group ) );
                std::get<2>( layers.back().second ).load( parameters[ node.input( 1 ) ], parameters[ node.input( 2 ) ] );
                std::cout << fmt::format( "ConvTranspose2d({}, {}, {}, {}, {})", in_channels, out_channels, kernel_shape, strides, pads, dilations ) << std::endl;
            } else if ( op_type == "InstanceNormalization" ) {
                int in_channels = 0;
                std::cout << node.input_size() << std::endl;
                std::cout << node.input( 1 ) << std::endl;
                std::cout << node.input( 2 ) << std::endl;
                auto s = node.input( 1 );

                if ( auto it = std::find_if( graph.initializer().begin(), graph.initializer().end(), [ s ]( auto &el ) { return el.name() == s; } ); it != graph.initializer().end() ) {
                    in_channels = it->dims( 0 );
                }
                T epsilon = 1e-5;
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "epsilon" )
                        epsilon = attribute.f();
                }
                layers.emplace_back( name, torch::InstanceNorm2d<T>( in_channels, epsilon ) );
                std::cout << fmt::format( "InstanceNorm2d (in_channels : {}, epsilon : {})", in_channels, epsilon ) << std::endl;
                std::get<3>( layers.back().second ).load( parameters[ node.input( 1 ) ], parameters[ node.input( 2 ) ] );
                std::cout << fmt::format( "InstanceNorm2d (in_channels : {}, epsilon : {})", in_channels, epsilon ) << std::endl;
            } else if ( op_type == "LeakyRelu" ) {
                T alpha = 0.0f;
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "alpha" )
                        alpha = attribute.f();
                }
                layers.emplace_back( name, torch::LeakyReLU<T>( alpha ) );
                std::cout << fmt::format( "LeakyReLU (alpha : {})", alpha ) << std::endl;
            } else if ( op_type == "MaxPool" ) {
                int kernel_shape, pads, strides;
                kernel_shape = pads = strides = 0;
                for ( int j = 0; j < node.attribute_size(); ++j ) {
                    auto attribute = node.attribute( j );
                    if ( attribute.name() == "kernel_shape" )
                        kernel_shape = attribute.ints( 0 );
                    else if ( attribute.name() == "pads" )
                        pads = attribute.ints( 0 );
                    else if ( attribute.name() == "strides" )
                        strides = attribute.ints( 0 );
                }

                layers.emplace_back( name, torch::MaxPool2d<T>( kernel_shape, strides, pads ) );
            } else if ( op_type == "Relu" )
                layers.emplace_back( name, torch::ReLU<T>() );
            else if ( op_type == "Tanh" )
                layers.emplace_back( name, torch::Tanh<T>() );
            else if ( op_type == "Add" )
                layers.emplace_back( name, torch::Add<T>() );
            else if ( op_type == "Identity" )
                layers.emplace_back( name, torch::Identity<T>() );
            else if ( op_type == "GlobalAveragePool" )
                layers.emplace_back( name, torch::GAP<T>() );
            else if ( op_type == "Flatten" )
                layers.emplace_back( name, torch::Flatten<T>() );
            else if ( op_type == "Gemm" ) {
                int in_dim, out_dim;
                in_dim = out_dim = 0;
                auto s = node.input( 1 );
                if ( auto it = std::find_if( graph.initializer().begin(), graph.initializer().end(), [ s ]( auto &el ) { return el.name() == s; } ); it != graph.initializer().end() ) {
                    in_dim = it->dims( 1 );
                    out_dim = it->dims( 0 );
                }
                layers.emplace_back( name, torch::Gemm<T>( in_dim, out_dim ) );
                std::get<12>( layers.back().second ).load( parameters[ node.input( 1 ) ], parameters[ node.input( 2 ) ] );
            } else
                std::cerr << "unknown op_type : " << op_type << std::endl;
        }
    }

    Tensor1D forward( Tensor3D &input ) {
        Tensor1D final_output;
        inputs.insert( { graph.input( 0 ).name(), input } );
        for ( int idx = 0; idx < graph.node_size(); ++idx ) {
            auto &[ name, node ] = layers[ idx ];
            if ( node.index() == 0 ) { // Conv
                auto &conv = std::get<0>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = conv( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 1 ) { // BatchNorm2d
                auto &batchnorm2d = std::get<1>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = batchnorm2d( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 2 ) { // ConvTranspose2d
                auto &convtranspose2d = std::get<2>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = convtranspose2d( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 3 ) { // InstanceNorm2d
                auto &instancenorm2d = std::get<3>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = instancenorm2d( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 4 ) { // LeakyReLU
                auto &lrelu = std::get<4>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = lrelu( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 5 ) { // MaxPool2d
                auto &maxpool = std::get<5>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = maxpool( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 6 ) { // ReLU
                auto &relu = std::get<6>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = relu( input );
                inputs.insert( { graph.node( idx ).output( 0 ), std::move( output ) } );
            } else if ( node.index() == 7 ) { // Tanh
                auto &tanh = std::get<7>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = tanh( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 8 ) { // Add
                auto &add = std::get<8>( node );
                auto &input1 = inputs[ graph.node( idx ).input( 0 ) ];
                auto &input2 = inputs[ graph.node( idx ).input( 1 ) ];
                auto output = add( input1, input2 );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 9 ) { // Identity
                auto &identity = std::get<9>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = identity( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 10 ) { // GAP
                auto &gap = std::get<10>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = gap( input );
                inputs.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 11 ) { // Flatten
                auto &flatten = std::get<11>( node );
                auto &input = inputs[ graph.node( idx ).input( 0 ) ];
                auto output = flatten( input );
                inputs_1D.insert( { graph.node( idx ).output( 0 ), output } );
            } else if ( node.index() == 12 ) { // Gemm
                auto &gemm = std::get<12>( node );
                auto &input = inputs_1D[ graph.node( idx ).input( 0 ) ];
                auto output = gemm( input );
                inputs_1D.insert( { graph.node( idx ).output( 0 ), output } );
            }
        }
        return inputs_1D[ graph.output( 0 ).name() ];
    }

    Tensor1D operator()( Tensor3D &input ) {
        auto output = forward( input );
        return output;
    }

private:
    onnx::ModelProto model;
    onnx::GraphProto graph;

    // vector of layers, form of {layer name, layer class}
    std::vector<std::pair<std::string, layer<T>>> layers;

    // map of {tensor name : 1D Tensor value}
    std::map<std::string, Tensor1D> inputs_1D;

    // map of {tensor name : 3D Tensor value}
    std::map<std::string, Tensor3D> inputs;
};