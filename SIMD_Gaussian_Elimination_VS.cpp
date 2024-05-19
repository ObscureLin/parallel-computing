// SIMD_Gaussian_Elimination_VS.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iomanip>
#include<windows.h>
#include <time.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512

#define N 500
using namespace std;

float m1[N][N];
float m2[N][N];

void m_reset() {
	srand(time(0));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float temp = rand() % 100;
			m1[i][j] = temp;
			m2[i][j] = temp;
		}
	}
}

// 1.首先将本行第一个元素(k,k)置为1，本行其他元素除以首元素
// 2.将本行之后的行进行先除再减，将(i,k)置零
void gauss_elim() {
	// 当前第k行
	for (int k = 0; k < N; k++) {
		for (int j = k + 1; j < N; j++) {
			m1[k][j] /= m1[k][k];
		}
		m1[k][k] = 1.0;
		for (int i = k + 1; i < N; i++) {
			for (int j = k + 1; j < N; j++) {
				m1[i][j] -= m1[i][k] * m1[k][j];
			}
			m1[i][k] = 0;
		}
	}
}

// 1.首先将本行第一个元素(k,k)置为1，本行其他元素除以首元素
// 2.将本行之后的行进行先除再减，将(i,k)置零
void parallel_gauss_elim() {
	// 当前第k行
	for (int k = 0; k < N; k++) {
		// 加载k行第一个元素
		__m128 vt = _mm_set1_ps(m2[k][k]);
		int j = 0;
		// 遍历本行但在k列之后的元素
		for (j = k + 1; j + 4 <= N; j += 4) {
			__m128 va = _mm_loadu_ps(&m2[k][j]);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(&m2[k][j], va);
		}
		// 本行末尾还剩几个元素
		for (; j < N; j++) {
			m2[k][j] = m2[k][j] / m2[k][k];
		}
		// 本行首个元素置1
		m2[k][k] = 1.0;

		// 对k行之后的行进行消除，形成下三角矩阵
		for (int i = k + 1; i < N; i++) {
			// (i,k)元素
			__m128 vaik = _mm_set1_ps(m2[i][k]);
			// 遍历i行，k+1列之后的元素
			for (j = k + 1; j + 4 <= N; j += 4) {
				__m128 vakj = _mm_loadu_ps(&m2[k][j]);
				__m128 vaij = _mm_loadu_ps(&m2[i][j]);
				__m128 vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(&m2[i][j], vaij);
			}
			// i行末尾还剩几个元素
			for (; j < N; j++) {
				m2[i][j] = m2[i][j] - m2[k][j] * m2[i][k];
			}
			m2[i][k] = 0;
		}

	}
}

int main()
{
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	m_reset();
	/*
	cout << "Raw matrix:" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(10) << m1[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl << "--------------------------------------" << endl << endl;
	*/

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	gauss_elim();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "Normal time:" << (tail - head) * 1000 / freq << "ms" << endl;
	cout << "------------------" << endl;

	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	parallel_gauss_elim();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "SSE time:" << (tail - head) * 1000 / freq << "ms" << endl;
	cout << "------------------" << endl;

	/*
	cout << "Normal result matrix:" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(10) << m1[i][j] << " ";
		}
		cout << endl;
	}

	cout << "Parallel result matrix:" << endl;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << setw(10) << m2[i][j] << " ";
		}
		cout << endl;
	}
	*/

}
