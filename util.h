#pragma once

#include <bits/stdc++.h>

using namespace std;

template <typename SourceType, typename TransformType>
auto compose(SourceType dataset, const TransformType &transform)
{
    return dataset.map(transform);
}
// base case of compose template

template <typename SourceType, typename TransformType, typename... TransformTypes>
auto compose(SourceType dataset, const TransformType &transform, const TransformTypes &...transforms)
{
    return compose(std::move(dataset.map(transform)), transforms...);
}
// variadic template for composition
