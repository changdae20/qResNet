#ifndef UTIL_HPP
#define UTIL_HPP

#include "onnx.proto3.pb.h"
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace UTIL {
template <typename T>
std::vector<std::vector<std::vector<T>>> img2vec( cv::Mat &image, std::map<std::string, std::array<float, 3>> dict ) {
    std::vector<std::vector<std::vector<T>>> input;
    input.resize( 3 );
    for ( auto &el : input ) {
        el.resize( image.rows );
        for ( auto &el2 : el ) {
            el2.resize( image.cols );
        }
    }

    std::cout << fmt::format( "[util.hpp L{}] image.rows : {}, image.cols : {}", __LINE__, image.rows, image.cols ) << std::endl;

    for ( int i = 0; i < image.rows; ++i ) {
        for ( int j = 0; j < image.cols; ++j ) {
            input[ 0 ][ i ][ j ] = T( ( ( image.at<cv::Vec3b>( i, j )[ 0 ] / 255.0f ) - dict[ "mean" ][ 0 ] ) / dict[ "std" ][ 0 ] );
            input[ 1 ][ i ][ j ] = T( ( ( image.at<cv::Vec3b>( i, j )[ 1 ] / 255.0f ) - dict[ "mean" ][ 1 ] ) / dict[ "std" ][ 1 ] );
            input[ 2 ][ i ][ j ] = T( ( ( image.at<cv::Vec3b>( i, j )[ 2 ] / 255.0f ) - dict[ "mean" ][ 2 ] ) / dict[ "std" ][ 2 ] );
        }
    }

    return input;
}

template <typename T>
cv::Mat vec2img( std::vector<std::vector<std::vector<T>>> &output ) {
    cv::Mat image( output[ 0 ].size(), output[ 0 ][ 0 ].size(), CV_8UC3 );

    for ( int i = 0; i < image.rows; ++i ) {
        for ( int j = 0; j < image.cols; ++j ) {
            image.at<cv::Vec3b>( i, j )[ 0 ] = std::clamp( static_cast<int>( ( static_cast<float>( output[ 0 ][ i ][ j ] ) + 1.0 ) * 255.0 / 2.0 ), 0, 255 );
            image.at<cv::Vec3b>( i, j )[ 1 ] = std::clamp( static_cast<int>( ( static_cast<float>( output[ 1 ][ i ][ j ] ) + 1.0 ) * 255.0 / 2.0 ), 0, 255 );
            image.at<cv::Vec3b>( i, j )[ 2 ] = std::clamp( static_cast<int>( ( static_cast<float>( output[ 2 ][ i ][ j ] ) + 1.0 ) * 255.0 / 2.0 ), 0, 255 );
        }
    }

    return image;
}

void write_txt( std::vector<std::vector<std::vector<float>>> &output ) {
    std::ofstream of( "output_cpp.txt" );
    for ( auto &el : output ) {
        for ( auto &el2 : el ) {
            for ( auto &el3 : el2 ) {
                of << el3 << '\n';
            }
        }
    }
    return;
}

void print_dim( const ::onnx::TensorShapeProto_Dimension &dim ) {
    switch ( dim.value_case() ) {
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimParam:
        std::cout << dim.dim_param();
        break;
    case onnx::TensorShapeProto_Dimension::ValueCase::kDimValue:
        std::cout << dim.dim_value();
        break;
    default:
        assert( false && "should never happen" );
    }
}

void print_io_info( const ::google::protobuf::RepeatedPtrField<::onnx::ValueInfoProto> &info ) {
    for ( auto input_data : info ) {
        auto shape = input_data.type().tensor_type().shape();
        std::cout << "  " << input_data.name() << ":";
        std::cout << "[";
        if ( shape.dim_size() != 0 ) {
            int size = shape.dim_size();
            for ( int i = 0; i < size - 1; ++i ) {
                print_dim( shape.dim( i ) );
                std::cout << ", ";
            }
            print_dim( shape.dim( size - 1 ) );
        }
        std::cout << "]\n";
    }
}

};     // namespace UTIL
#endif // UTIL_HPP