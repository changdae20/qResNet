#include "AFloat.hpp"
#include "CatGenerator.hpp"
#include "labels.hpp"
#include "onnx.hpp"
#include "onnx.proto3.pb.h"
#include "resnet.hpp"
#include "util.hpp"
#include <iostream>
#include <map>
#include <string>
#include <variant>

extern std::map<int, std::string> id2label;
int main( int argc, char **argv ) {
    constexpr int m = 11;
    constexpr int e = 4;
    mpfr_set_emax( ( 1 << ( e - 1 ) ) - 1 );
    mpfr_set_emin( -( 1 << ( e - 1 ) ) );
    auto model = Onnx<AFloat<m>>( "resnet-18.onnx" );

    auto start = std::chrono::high_resolution_clock::now();
    auto image = cv::imread( "tiger128.jpg" );

    auto input = UTIL::img2vec<AFloat<m>>( image, { { "mean", { 0.485, 0.456, 0.406 } },
                                                    { "std", { 0.229, 0.224, 0.225 } } } );

    auto output = model( input );

    std::vector<std::pair<AFloat<m>, int>> output_with_index;
    for ( int i = 0; i < output.size(); ++i ) {
        output_with_index.push_back( std::pair( output[ i ], i ) );
    }
    std::sort( output_with_index.begin(), output_with_index.end(), []( auto &left, auto &right ) {
        return left.first > right.first;
    } );
    std::cout << fmt::format( "Current Precision : total {}-bit", ( m + e ) ) << std::endl;
    std::cout << "=== Predicted Result ===" << std::endl;
    for ( int i = 0; i < 5; ++i ) {
        std::cout << fmt::format( "Top {} : {} with logit {}", i + 1, id2label[ output_with_index[ i ].second ], static_cast<std::string>( output_with_index[ i ].first ) ) << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time : " << std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count() << "ms" << std::endl;

    return 0;
}