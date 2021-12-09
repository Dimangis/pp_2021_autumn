// Copyright 2021 Lebedev Alexey
#include <gtest/gtest.h>
#include <string>
#include <random>
#include "./jacobi_iterations.h"
#include <gtest-mpi-listener.hpp>

#define eps 0.0001

TEST(Jacobi_iterations_MPI, Test_matmul2D) {
    Tensor<int> t1({2,4});
    Tensor<int> t2({4,3});
    for (size_t i = 0; i < t1.get_size(); i++) {
        t1[i] = i + 1;
    }
    for (size_t i = 0; i < t2.get_size(); i++) {
        t2[i] = i + 1;
    }

    //               1  2  3
    // 1 2 3 4   X   4  5  6   =    70  80  90
    // 5 6 7 8       7  8  9       158 184 210
    //               10 11 12
    Tensor<int> expected({2,3});
    expected[0] = 70, expected[1] = 80, expected[2] = 90;
    expected[3] = 158, expected[4] = 184; expected[5] = 210;

    Tensor<int> result = matmul2D(t1, t2);

    ASSERT_EQ(result.get_shape(), expected.get_shape());
    for (size_t i = 0; i < result.get_size(); i++) {
        ASSERT_EQ(result[i], expected[i]);
    }
}

TEST(Jacobi_iterations_MPI, Test_Sequential) {
    LinearSystem<int> lin_sys_int(3);
    std::vector<int> A1 = {10, 1, -1, 1, 10, -1, -1, 1, 10};
    std::vector<int> b1 = {11, 10, 10};
    std::vector<int> x1 = {11, 10, 10};
    std::memcpy(lin_sys_int.A.get_data(), A1.data(), lin_sys_int.A.get_size() * sizeof(int));
    std::memcpy(lin_sys_int.b.get_data(), b1.data(), lin_sys_int.b.get_size() * sizeof(int));
    std::memcpy(lin_sys_int.x0.get_data(), x1.data(), lin_sys_int.x0.get_size() * sizeof(int));
    Tensor<double> x = jacobi_iterations_sequential(lin_sys_int, eps);
    EXPECT_NEAR(x[0], 1.10202, eps);
    EXPECT_NEAR(x[1], 0.99091, eps);
    EXPECT_NEAR(x[2], 1.01111, eps);

    LinearSystem<double> lin_sys_double(3);
    std::vector<double> A2 = {9.2, 2.5, -3.7, 0.9, 9, 0.2, 4.5, -1.6, -10.3};
    std::vector<double> b2 = {-17.5, 4.4, -22.1};
    std::vector<double> x2 = {0, 0, 0};
    std::memcpy(lin_sys_double.A.get_data(), A2.data(), lin_sys_double.A.get_size() * sizeof(double));
    std::memcpy(lin_sys_double.b.get_data(), b2.data(), lin_sys_double.b.get_size() * sizeof(double));
    std::memcpy(lin_sys_double.x0.get_data(), x2.data(), lin_sys_double.x0.get_size() * sizeof(double));
    x = jacobi_iterations_sequential(lin_sys_double, eps);
    EXPECT_NEAR(x[0], -1.50757, eps);
    EXPECT_NEAR(x[1], 0.60870, eps);
    EXPECT_NEAR(x[2], 1.39242, eps);
}

TEST(Jacobi_iterations_MPI, Test_Parallel_Random_Data_Same_Size) {
}

TEST(Jacobi_iterations_MPI, Test_Parallel_Random_Data_Different_Size) {
}

TEST(Jacobi_iterations_MPI, Test_Parallel_Random_Data_Random_Size) {
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}