#ifndef AFLOAT_HPP
#define AFLOAT_HPP

#include <cmath>
#include <gmp.h>
#include <iostream>
#include <mpfr.h>
#include <string>

template <int P>
class AFloat {
public:
    AFloat() {
        mpfr_init2( value, P );
    }
    AFloat( float d ) {
        mpfr_init2( value, P );
        mpfr_set_flt( value, d, GMP_RNDD );
    }
    AFloat( double d ) {
        mpfr_init2( value, P );
        mpfr_set_d( value, d, GMP_RNDD );
    }
    AFloat( const AFloat<P> &other ) {
        mpfr_init2( value, P );
        mpfr_set( value, other.value, GMP_RNDD );
    }
    AFloat( AFloat<P> &&other ) {
        mpfr_init2( value, P );
        mpfr_set( value, other.value, GMP_RNDD );
    }

    ~AFloat() {
        mpfr_clear( value );
    }

    std::string to_string() {
        char c[ 256 ];
        mpfr_sprintf( c, "%.128Rf", value );
        return std::string( c );
    }

    inline AFloat<P> operator*( const AFloat<P> &other ) {
        AFloat<P> ans;
        mpfr_mul( ans.value, value, other.value, GMP_RNDU );
        return ans;
    }

    inline AFloat<P> operator+( const AFloat<P> &other ) {
        AFloat<P> ans;
        mpfr_add( ans.value, value, other.value, GMP_RNDU );
        return ans;
    }

    inline AFloat<P> operator-( const AFloat<P> &other ) {
        AFloat<P> ans;
        mpfr_sub( ans.value, value, other.value, GMP_RNDU );
        return ans;
    }

    inline AFloat<P> operator/( const AFloat<P> &other ) {
        AFloat<P> ans;
        mpfr_div( ans.value, value, other.value, GMP_RNDU );
        return ans;
    }

    inline AFloat<P> &operator=( const float d ) {
        mpfr_set_flt( value, d, GMP_RNDD );
        return *this;
    }

    inline AFloat<P> &operator=( const double d ) {
        mpfr_set_d( value, d, GMP_RNDD );
        return *this;
    }

    inline AFloat<P> &operator=( const AFloat<P> &other ) {
        if ( this == &other )
            return *this;

        mpfr_set( value, other.value, GMP_RNDD );
        return *this;
    }

    inline AFloat<P> &operator+=( const AFloat<P> &other ) {
        mpfr_add( value, value, other.value, GMP_RNDU );
        return *this;
    }

    inline AFloat<P> &operator*=( const AFloat<P> &other ) {
        mpfr_mul( value, value, other.value, GMP_RNDU );
        return *this;
    }

    inline AFloat<P> &operator/=( const AFloat<P> &other ) {
        mpfr_div( value, value, other.value, GMP_RNDU );
        return *this;
    }

    inline bool operator>( const AFloat<P> &other ) {
        return mpfr_greater_p( value, other.value );
    }

    inline bool operator<( const AFloat<P> &other ) {
        return mpfr_less_p( value, other.value );
    }

    inline operator float() const {
        return mpfr_get_flt( value, GMP_RNDD );
    }

    inline operator std::string() const {
        char c[ 256 ];
        mpfr_sprintf( c, "%.20Rf", value );
        return std::string( c );
    }

    inline AFloat<P> &operator++() {
        mpfr_nextabove( value );
        return *this;
    }

public:
    mpfr_t value;
};

template <>
class AFloat<32> {
public:
    AFloat() {
        value = 0.0f;
    }
    AFloat( double d ) {
        value = static_cast<float>( d );
    }

    ~AFloat() {}

    std::string to_string() {
        return std::to_string( value );
    }

    AFloat<32> operator*( const AFloat<32> &other ) {
        AFloat<32> ans;
        ans.value = value * other.value;
        return ans;
    }

    AFloat<32> operator+( const AFloat<32> &other ) {
        AFloat<32> ans;
        ans.value = value + other.value;
        return ans;
    }

    AFloat<32> operator-( const AFloat<32> &other ) {
        AFloat<32> ans;
        ans.value = value - other.value;
        return ans;
    }

    AFloat<32> operator/( const AFloat<32> &other ) {
        AFloat<32> ans;
        ans.value = value / other.value;
        return ans;
    }

    AFloat<32> &operator=( const double d ) {
        value = static_cast<float>( d );
        return *this;
    }

    AFloat<32> &operator=( const AFloat<32> &other ) {
        if ( this == &other )
            return *this;

        value = other.value;
        return *this;
    }

    AFloat<32> &operator+=( const AFloat<32> &other ) {
        value += other.value;
        return *this;
    }

    AFloat<32> &operator*=( const AFloat<32> &other ) {
        value *= other.value;
        return *this;
    }

    AFloat<32> &operator/=( const AFloat<32> &other ) {
        value /= other.value;
        return *this;
    }

    bool operator>( const AFloat<32> &other ) {
        return value > other.value;
    }

    bool operator<( const AFloat<32> &other ) {
        return value < other.value;
    }

    operator float() const {
        return value;
    }

    inline operator std::string() const {
        return std::to_string( value );
    }

public:
    float value;
};

template <int P>
AFloat<P> sqrt( const AFloat<P> &other ) {
    AFloat<P> ans;
    mpfr_sqrt( ans.value, other.value, GMP_RNDU );
    return ans;
}

template <>
AFloat<32> sqrt( const AFloat<32> &other ) {
    AFloat<32> ans;
    ans.value = std::sqrt( other.value );
    return ans;
}

template <int P>
AFloat<P> tanh( const AFloat<P> &other ) {
    AFloat<P> ans;
    mpfr_tanh( ans.value, other.value, GMP_RNDU );
    return ans;
}

template <>
AFloat<32> tanh( const AFloat<32> &other ) {
    AFloat<32> ans;
    ans.value = std::tanh( other.value );
    return ans;
}

#endif // AFLOAT_HPP