// Copyright 2021 Lebedev Alexey
#include <gtest/gtest.h>
#include <cstring>
#include "./jacobi_iterations.h"
#include <gtest-mpi-listener.hpp>

#define eps 0.0001


TEST(Jacobi_iterations_MPI, Test_Tensor) {
    std::vector<size_t> shape = {1, 3, 28, 28};
    std::vector<size_t> strides = {2352, 784, 28, 1};
    size_t size = 2352;
    Tensor<int> t(shape);
    ASSERT_EQ(t.get_shape(), shape);
    ASSERT_EQ(t.get_strides(), strides);
    ASSERT_EQ(t.get_size(), size);
    ASSERT_TRUE(t.is_allocated());

    Tensor<int> t_copy = t;
    t_copy[0] = 1;
    // shares memory
    ASSERT_EQ(t[0], 1);

    Tensor<int> empty_t;
    ASSERT_FALSE(empty_t.is_allocated());
    empty_t.resize({1, 5});
    ASSERT_TRUE(empty_t.is_allocated());
}


TEST(Jacobi_iterations_MPI, Test_matmul2D) {
    Tensor<int> t1({2, 4});
    Tensor<int> t2({4, 3});

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

    Tensor<int> expected({2, 3});
    expected[0] = 70, expected[1] = 80, expected[2] = 90;
    expected[3] = 158, expected[4] = 184; expected[5] = 210;

    Tensor<int> result = matmul2D(t1, t2);

    ASSERT_EQ(result.get_shape(), expected.get_shape());
    for (size_t i = 0; i < result.get_size(); i++) {
        ASSERT_EQ(result[i], expected[i]);
    }
}


TEST(Jacobi_iterations_MPI, Test_Jacobi_Iterations_Sequential) {
    LinearSystem sys1(3);
    std::vector<float> A1 = {10, 1, -1, 1, 10, -1, -1, 1, 10};
    std::vector<float> b1 = {11, 10, 10};
    std::vector<float> x1 = {11, 10, 10};
    std::memcpy(sys1.A.get_data(), A1.data(), sys1.A.get_size() * sizeof(float));
    std::memcpy(sys1.b.get_data(), b1.data(), sys1.b.get_size() * sizeof(float));
    std::memcpy(sys1.x0.get_data(), x1.data(), sys1.x0.get_size() * sizeof(float));
    Tensor<float> x = sys1.solve(eps);
    EXPECT_NEAR(x[0], 1.10202, eps);
    EXPECT_NEAR(x[1], 0.99091, eps);
    EXPECT_NEAR(x[2], 1.01111, eps);

    LinearSystem sys2(3);
    std::vector<float> A2 = {9.2, 2.5, -3.7, 0.9, 9, 0.2, 4.5, -1.6, -10.3};
    std::vector<float> b2 = {-17.5, 4.4, -22.1};
    std::vector<float> x2 = {0, 0, 0};
    std::memcpy(sys2.A.get_data(), A2.data(), sys2.A.get_size() * sizeof(float));
    std::memcpy(sys2.b.get_data(), b2.data(), sys2.b.get_size() * sizeof(float));
    std::memcpy(sys2.x0.get_data(), x2.data(), sys2.x0.get_size() * sizeof(float));
    x = sys2.solve(eps);
    EXPECT_NEAR(x[0], -1.50757, eps);
    EXPECT_NEAR(x[1], 0.60870, eps);
    EXPECT_NEAR(x[2], 1.39242, eps);
}


TEST(Jacobi_iterations_MPI, Test_Jacobi_Iterations_Parallel) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n_dims = 0;

    if (rank == 0) {
        n_dims = 3;
    }

    LinearSystem sys1(n_dims);

    if (rank == 0) {
        std::vector<float> A1 = {10, 1, -1, 1, 10, -1, -1, 1, 10};
        std::vector<float> b1 = {11, 10, 10};
        std::vector<float> x1 = {11, 10, 10};
        std::memcpy(sys1.A.get_data(), A1.data(), sys1.A.get_size() * sizeof(float));
        std::memcpy(sys1.b.get_data(), b1.data(), sys1.b.get_size() * sizeof(float));
        std::memcpy(sys1.x0.get_data(), x1.data(), sys1.x0.get_size() * sizeof(float));
    }

    Tensor<float> x = solve_parallel(sys1, eps);

    if (rank == 0) {
        EXPECT_NEAR(x[0], 1.10202, eps);
        EXPECT_NEAR(x[1], 0.99091, eps);
        EXPECT_NEAR(x[2], 1.01111, eps);
    } else if (rank >= 3) {
        ASSERT_FALSE(x.is_allocated());
    }

    /*LinearSystem sys2(n_dims);

    if (rank == 0) {
        std::vector<float> A2 = {9.2, 2.5, -3.7, 0.9, 9, 0.2, 4.5, -1.6, -10.3};
        std::vector<float> b2 = {-17.5, 4.4, -22.1};
        std::vector<float> x2 = {0, 0, 0};
        std::memcpy(sys2.A.get_data(), A2.data(), sys2.A.get_size() * sizeof(float));
        std::memcpy(sys2.b.get_data(), b2.data(), sys2.b.get_size() * sizeof(float));
        std::memcpy(sys2.x0.get_data(), x2.data(), sys2.x0.get_size() * sizeof(float));
    }

    x = solve_parallel(sys2, eps);

    if (rank == 0) {
        EXPECT_NEAR(x[0], -1.50757, eps);
        EXPECT_NEAR(x[1], 0.60870, eps);
        EXPECT_NEAR(x[2], 1.39242, eps);
    } else if (rank >= 3) {
        ASSERT_FALSE(x.is_allocated());
    }*/
}


TEST(Jacobi_iterations_MPI, Test_Jacobi_Iterations_Parallel_4D) {
    /*int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n_dims = 0;

    if (rank == 0) {
        n_dims = 4;
    }

    LinearSystem sys(n_dims);
    if (rank == 0) {
        std::vector<float> A = {20.9, 1.2, 2.1, 0.9, 1.2, 21.2, 1.5, 2.5, 2.1, 1.5, 19.8, 1.3, 0.9, 2.5, 1.3, 32.1};
        std::vector<float> b = {21.7, 27.46, 28.76, 49.72};
        std::vector<float> x0 = {0, 0, 0, 0};
        std::memcpy(sys.A.get_data(), A.data(), sys.A.get_size() * sizeof(float));
        std::memcpy(sys.b.get_data(), b.data(), sys.b.get_size() * sizeof(float));
        std::memcpy(sys.x0.get_data(), x0.data(), sys.x0.get_size() * sizeof(float));
    }

    Tensor<float> x = solve_parallel(sys, eps);

    if (rank == 0) {
        Tensor<float> expected = sys.solve(eps);
        for (size_t i = 0; i < x.get_size(); i++) {
            EXPECT_NEAR(x[i], expected[i], eps);
        }
    } else if (rank >= 4) {
        ASSERT_FALSE(x.is_allocated());
    }*/
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
