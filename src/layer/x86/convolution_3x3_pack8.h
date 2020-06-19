// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.


static inline __m256 vaddq_f32(__m256 a, __m256 b){
    return _mm256_add_ps(a,b);
}
static inline __m256 vsubq_f32(__m256 a, __m256 b){
    return _mm256_sub_ps(a,b);
}
static inline __m256 vmulq_f32(__m256 a, __m256 b){
    return _mm256_mul_ps(a,b);
}
static inline __m256 vmulq_n_f32(__m256 a, float b){
    return _mm256_mul_ps(a,_mm256_set1_ps(b));
}
static inline __m256 vmlaq_n_f32(__m256 a, __m256 b, float c){
    return _mm256_add_ps(a,_mm256_mul_ps(b,_mm256_set1_ps(c)));
}

static inline __m256 vmlsq_n_f32(__m256 a, __m256 b, float c){
    return _mm256_sub_ps(a,_mm256_mul_ps(b,_mm256_set1_ps(c)));
}



static void conv3x3s1_winograd64_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch)
{
    // winograd63 transform kernel
    Mat kernel_tm;
    kernel_tm.create(8 * 8, inch, outch);

    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i = 0; i < 8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j = 0; j < 8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++)
                {
                    kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // interleave
    // src = 64-inch-outch
    // dst = 4b-4a-inch/4a-64-outch/4b;
    kernel_tm_pack8.create(2 * inch / 8, 64, (outch / 4) / 2 + (outch / 4) % 2, (size_t)4u * 64, 64);
    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 3 < inch; p += 4)
            {
                const float* k00 = k0.row(p);
                const float* k01 = k0.row(p + 1);
                const float* k02 = k0.row(p + 2);
                const float* k03 = k0.row(p + 3);
                const float* k04 = k0.row(p + 4);
                const float* k05 = k0.row(p + 5);
                const float* k06 = k0.row(p + 6);
                const float* k07 = k0.row(p + 7);

                const float* k10 = k1.row(p);
                const float* k11 = k1.row(p + 1);
                const float* k12 = k1.row(p + 2);
                const float* k13 = k1.row(p + 3);
                const float* k14 = k1.row(p + 4);
                const float* k15 = k1.row(p + 5);
                const float* k16 = k1.row(p + 6);
                const float* k17 = k1.row(p + 7);

                const float* k20 = k2.row(p);
                const float* k21 = k2.row(p + 1);
                const float* k22 = k2.row(p + 2);
                const float* k23 = k2.row(p + 3);
                const float* k24 = k2.row(p + 4);
                const float* k25 = k2.row(p + 5);
                const float* k26 = k2.row(p + 6);
                const float* k27 = k2.row(p + 7);

                const float* k30 = k3.row(p);
                const float* k31 = k3.row(p + 1);
                const float* k32 = k3.row(p + 2);
                const float* k33 = k3.row(p + 3);
                const float* k34 = k3.row(p + 4);
                const float* k35 = k3.row(p + 5);
                const float* k36 = k3.row(p + 6);
                const float* k37 = k3.row(p + 7);

                const float* k40 = k4.row(p);
                const float* k41 = k4.row(p + 1);
                const float* k42 = k4.row(p + 2);
                const float* k43 = k4.row(p + 3);
                const float* k44 = k4.row(p + 4);
                const float* k45 = k4.row(p + 5);
                const float* k46 = k4.row(p + 6);
                const float* k47 = k4.row(p + 7);

                const float* k50 = k5.row(p);
                const float* k51 = k5.row(p + 1);
                const float* k52 = k5.row(p + 2);
                const float* k53 = k5.row(p + 3);
                const float* k54 = k5.row(p + 4);
                const float* k55 = k5.row(p + 5);
                const float* k56 = k5.row(p + 6);
                const float* k57 = k5.row(p + 7);

                const float* k60 = k6.row(p);
                const float* k61 = k6.row(p + 1);
                const float* k62 = k6.row(p + 2);
                const float* k63 = k6.row(p + 3);
                const float* k64 = k6.row(p + 4);
                const float* k65 = k6.row(p + 5);
                const float* k66 = k6.row(p + 6);
                const float* k67 = k6.row(p + 7);

                const float* k70 = k7.row(p);
                const float* k71 = k7.row(p + 1);
                const float* k72 = k7.row(p + 2);
                const float* k73 = k7.row(p + 3);
                const float* k74 = k7.row(p + 4);
                const float* k75 = k7.row(p + 5);
                const float* k76 = k7.row(p + 6);
                const float* k77 = k7.row(p + 7);

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00[8] = k01[k];
                g00[9] = k11[k];
                g00[10] = k21[k];
                g00[11] = k31[k];

                g00[12] = k41[k];
                g00[13] = k51[k];
                g00[14] = k61[k];
                g00[15] = k71[k];

                g00[16] = k02[k];
                g00[17] = k12[k];
                g00[18] = k22[k];
                g00[19] = k32[k];

                g00[20] = k42[k];
                g00[21] = k52[k];
                g00[22] = k62[k];
                g00[23] = k72[k];

                g00[24] = k03[k];
                g00[25] = k13[k];
                g00[26] = k23[k];
                g00[27] = k33[k];

                g00[28] = k43[k];
                g00[29] = k53[k];
                g00[30] = k63[k];
                g00[31] = k73[k];

                g00 += 32;
            }
        }
    }
    fprintf(stderr, "TEST3\n");

}


static void conv3x3s1_winograd64_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    const float* bias = _bias;
    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);

        //         const float itm[8][8] = {
        //             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
        //
        //             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
        //
        //             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        //         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

            float tmp[8][8][8];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _r00 = _mm256_loadu_ps(r0);
                        __m256 _r01 = _mm256_loadu_ps(r0 + 8);
                        __m256 _r02 = _mm256_loadu_ps(r0 + 16);
                        __m256 _r03 = _mm256_loadu_ps(r0 + 24);
                        __m256 _r04 = _mm256_loadu_ps(r0 + 32);
                        __m256 _r05 = _mm256_loadu_ps(r0 + 40);
                        __m256 _r06 = _mm256_loadu_ps(r0 + 48);
                        __m256 _r07 = _mm256_loadu_ps(r0 + 56);

                        __m256 _tmp0m = vmlaq_n_f32(vsubq_f32(_r00, _r06), vsubq_f32(_r04, _r02), 5.25f);
                        __m256 _tmp7m = vmlaq_n_f32(vsubq_f32(_r07, _r01), vsubq_f32(_r03, _r05), 5.25f);
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[7][m], _tmp7m);

                        //                         tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25;
                        //                         tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25;

                        __m256 _tmp12a = vmlsq_n_f32(vaddq_f32(_r02, _r06), _r04, 4.25f);
                        __m256 _tmp12b = vmlsq_n_f32(vaddq_f32(_r01, _r05), _r03, 4.25f);

                        //                         float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25);
                        //                         float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25);

                        __m256 _tmp1m = vaddq_f32(_tmp12a, _tmp12b);
                        __m256 _tmp2m = vsubq_f32(_tmp12a, _tmp12b);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);

                        //                         tmp[1][m] = tmp12a + tmp12b;
                        //                         tmp[2][m] = tmp12a - tmp12b;

                        __m256 _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_r06, _r02, 0.25f), _r04, 1.25f);
                        __m256 _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 0.5f), _r03, 2.5f), _r05, 2.f);

                        //                         float tmp34a = (r0[6] + r0[2] * 0.25 - r0[4] * 1.25);
                        //                         float tmp34b = (r0[1] * 0.5 - r0[3] * 2.5 + r0[5] * 2);

                        __m256 _tmp3m = vaddq_f32(_tmp34a, _tmp34b);
                        __m256 _tmp4m = vsubq_f32(_tmp34a, _tmp34b);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
                        _mm256_storeu_ps(tmp[4][m], _tmp4m);

                        //                         tmp[3][m] = tmp34a + tmp34b;
                        //                         tmp[4][m] = tmp34a - tmp34b;

                        __m256 _tmp56a = vmlaq_n_f32(_r06, vmlsq_n_f32(_r02, _r04, 1.25f), 4.f);
                        __m256 _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_r01, 2.f), _r03, 2.5f), _r05, 0.5f);

                        //                         float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25) * 4);
                        //                         float tmp56b = (r0[1] * 2 - r0[3] * 2.5 + r0[5] * 0.5);

                        __m256 _tmp5m = vaddq_f32(_tmp56a, _tmp56b);
                        __m256 _tmp6m = vsubq_f32(_tmp56a, _tmp56b);
                        _mm256_storeu_ps(tmp[5][m], _tmp5m);
                        _mm256_storeu_ps(tmp[6][m], _tmp6m);

                        //                         tmp[5][m] = tmp56a + tmp56b;
                        //                         tmp[6][m] = tmp56a - tmp56b;

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 16;
                    float* r0_tm_3 = r0_tm_0 + tiles * 24;
                    float* r0_tm_4 = r0_tm_0 + tiles * 32;
                    float* r0_tm_5 = r0_tm_0 + tiles * 40;
                    float* r0_tm_6 = r0_tm_0 + tiles * 48;
                    float* r0_tm_7 = r0_tm_0 + tiles * 56;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_loadu_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_loadu_ps(tmp[m][7]);

                        __m256 _r0tm0 = vmlaq_n_f32(vsubq_f32(_tmp00, _tmp06), vsubq_f32(_tmp04, _tmp02), 5.25f);
                        __m256 _r0tm7 = vmlaq_n_f32(vsubq_f32(_tmp07, _tmp01), vsubq_f32(_tmp03, _tmp05), 5.25f);

                        //                         r0_tm[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25;
                        //                         r0_tm[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25;

                        __m256 _tmp12a = vmlsq_n_f32(vaddq_f32(_tmp02, _tmp06), _tmp04, 4.25f);
                        __m256 _tmp12b = vmlsq_n_f32(vaddq_f32(_tmp01, _tmp05), _tmp03, 4.25f);

                        //                         float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25);
                        //                         float tmp12b = (tmp0[1] + tmp0[5] - tmp0[3] * 4.25);

                        __m256 _r0tm1 = vaddq_f32(_tmp12a, _tmp12b);
                        __m256 _r0tm2 = vsubq_f32(_tmp12a, _tmp12b);

                        //                         r0_tm[1] = tmp12a + tmp12b;
                        //                         r0_tm[2] = tmp12a - tmp12b;

                        __m256 _tmp34a = vmlsq_n_f32(vmlaq_n_f32(_tmp06, _tmp02, 0.25f), _tmp04, 1.25f);
                        __m256 _tmp34b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 0.5f), _tmp03, 2.5f), _tmp05, 2.f);

                        //                         float tmp34a = (tmp0[6] + tmp0[2] * 0.25 - tmp0[4] * 1.25);
                        //                         float tmp34b = (tmp0[1] * 0.5 - tmp0[3] * 2.5 + tmp0[5] * 2);

                        __m256 _r0tm3 = vaddq_f32(_tmp34a, _tmp34b);
                        __m256 _r0tm4 = vsubq_f32(_tmp34a, _tmp34b);

                        //                         r0_tm[3] = tmp34a + tmp34b;
                        //                         r0_tm[4] = tmp34a - tmp34b;

                        __m256 _tmp56a = vmlaq_n_f32(_tmp06, vmlsq_n_f32(_tmp02, _tmp04, 1.25f), 4.f);
                        __m256 _tmp56b = vmlaq_n_f32(vmlsq_n_f32(vmulq_n_f32(_tmp01, 2.f), _tmp03, 2.5f), _tmp05, 0.5f);

                        //                         float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25) * 4);
                        //                         float tmp56b = (tmp0[1] * 2 - tmp0[3] * 2.5 + tmp0[5] * 0.5);

                        __m256 _r0tm5 = vaddq_f32(_tmp56a, _tmp56b);
                        __m256 _r0tm6 = vsubq_f32(_tmp56a, _tmp56b);

                        //                         r0_tm[5] = tmp56a + tmp56b;
                        //                         r0_tm[6] = tmp56a - tmp56b;

                        _mm256_storeu_ps(r0_tm_0, _r0tm0);
                        _mm256_storeu_ps(r0_tm_1, _r0tm1);
                        _mm256_storeu_ps(r0_tm_2, _r0tm2);
                        _mm256_storeu_ps(r0_tm_3, _r0tm3);
                        _mm256_storeu_ps(r0_tm_4, _r0tm4);
                        _mm256_storeu_ps(r0_tm_5, _r0tm5);
                        _mm256_storeu_ps(r0_tm_6, _r0tm6);
                        _mm256_storeu_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 64;
                        r0_tm_1 += tiles * 64;
                        r0_tm_2 += tiles * 64;
                        r0_tm_3 += tiles * 64;
                        r0_tm_4 += tiles * 64;
                        r0_tm_5 += tiles * 64;
                        r0_tm_6 += tiles * 64;
                        r0_tm_7 += tiles * 64;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input
        fprintf(stderr, "bottom_blob_tm = %d x %d x %d elempack = %d \n",bottom_blob_tm.w,bottom_blob_tm.h,bottom_blob_tm.c,bottom_blob_tm.elempack);

        fprintf(stderr, "bottom_blob_tm = %f %f %f %f %f %f %f %f\n", bottom_blob_tm[0],bottom_blob_tm[1],bottom_blob_tm[2],bottom_blob_tm[3],bottom_blob_tm[4],bottom_blob_tm[5],bottom_blob_tm[6],bottom_blob_tm[7]);
        fprintf(stderr, "bottom_blob_tm = %f %f %f %f %f %f %f %f\n", bottom_blob_tm[8],bottom_blob_tm[9],bottom_blob_tm[10],bottom_blob_tm[11],bottom_blob_tm[12],bottom_blob_tm[13],bottom_blob_tm[14],bottom_blob_tm[15]);

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = h_tm / 8 * w_tm / 8;
        fprintf(stderr, "2 tiles = %d \n",tiles);

        // permute
        //         bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;

        #if __aarch64__
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
#else
        if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                float* tm2p = tm2.row(i / 12);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r1 = _mm256_loadu_ps(r0+8);
                    __m256 _r2 = _mm256_loadu_ps(r0+16);
                    __m256 _r3 = _mm256_loadu_ps(r0+24);
                    __m256 _r4 = _mm256_loadu_ps(r0+32);
                    __m256 _r5 = _mm256_loadu_ps(r0+40);
                    __m256 _r6 = _mm256_loadu_ps(r0+48);
                    __m256 _r7 = _mm256_loadu_ps(r0+56);
                    __m256 _r8 = _mm256_loadu_ps(r0+64);
                    __m256 _r9 = _mm256_loadu_ps(r0+72);
                    __m256 _r10 = _mm256_loadu_ps(r0+80);
                    __m256 _r11 = _mm256_loadu_ps(r0+88);
                    _mm256_storeu_ps(tm2p,_r0);
                    _mm256_storeu_ps(tm2p+8,_r1);
                    _mm256_storeu_ps(tm2p+16,_r2);
                    _mm256_storeu_ps(tm2p+24,_r3);
                    _mm256_storeu_ps(tm2p+32,_r4);
                    _mm256_storeu_ps(tm2p+40,_r5);
                    _mm256_storeu_ps(tm2p+48,_r6);
                    _mm256_storeu_ps(tm2p+56,_r7);
                    _mm256_storeu_ps(tm2p+64,_r8);
                    _mm256_storeu_ps(tm2p+72,_r9);
                    _mm256_storeu_ps(tm2p+80,_r10);
                    _mm256_storeu_ps(tm2p+88,_r11);
                    tm2p += 96;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tm2p = tm2.row(i / 12 + (i % 12) / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r1 = _mm256_loadu_ps(r0+8);
                    __m256 _r2 = _mm256_loadu_ps(r0+16);
                    __m256 _r3 = _mm256_loadu_ps(r0+24);
                    __m256 _r4 = _mm256_loadu_ps(r0+32);
                    __m256 _r5 = _mm256_loadu_ps(r0+40);
                    __m256 _r6 = _mm256_loadu_ps(r0+48);
                    __m256 _r7 = _mm256_loadu_ps(r0+56);
                    _mm256_storeu_ps(tm2p,_r0);
                    _mm256_storeu_ps(tm2p+8,_r1);
                    _mm256_storeu_ps(tm2p+16,_r2);
                    _mm256_storeu_ps(tm2p+24,_r3);
                    _mm256_storeu_ps(tm2p+32,_r4);
                    _mm256_storeu_ps(tm2p+40,_r5);
                    _mm256_storeu_ps(tm2p+48,_r6);
                    _mm256_storeu_ps(tm2p+56,_r7);
                    tm2p += 64;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r1 = _mm256_loadu_ps(r0+8);
                    __m256 _r2 = _mm256_loadu_ps(r0+16);
                    __m256 _r3 = _mm256_loadu_ps(r0+24);
                    _mm256_storeu_ps(tm2p,_r0);
                    _mm256_storeu_ps(tm2p+8,_r1);
                    _mm256_storeu_ps(tm2p+16,_r2);
                    _mm256_storeu_ps(tm2p+24,_r3);
                    tm2p += 32;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    __m256 _r1 = _mm256_loadu_ps(r0+8);
                    _mm256_storeu_ps(tm2p,_r0);
                    _mm256_storeu_ps(tm2p+8,_r1);
                    tm2p += 16;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }

            for (; i < tiles; i++)
            {
                float* tm2p = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    _mm256_storeu_ps(tm2p,_r0);
                    tm2p += 8;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end
        fprintf(stderr, "bottom_blob_tm2 = %f %f %f %f %f %f %f %f\n", bottom_blob_tm2[0],bottom_blob_tm2[1],bottom_blob_tm2[2],bottom_blob_tm2[3],bottom_blob_tm2[4],bottom_blob_tm2[5],bottom_blob_tm2[6],bottom_blob_tm2[7]);
        fprintf(stderr, "bottom_blob_tm2 = %f %f %f %f %f %f %f %f\n", bottom_blob_tm2[8],bottom_blob_tm2[9],bottom_blob_tm2[10],bottom_blob_tm2[11],bottom_blob_tm2[12],bottom_blob_tm2[13],bottom_blob_tm2[14],bottom_blob_tm2[15]);

        top_blob_tm.create(tiles, 64, outch, elemsize, elempack, opt.workspace_allocator);

        int nn_outch = 0;
        int remain_outch_start = 0;

        //   "eor    v16.16b, v16.16b, v16.16b   \n"

        // "0:                                 \n"

        // "prfm   pldl1keep, [%2, #128]       \n"
        // "ld1    {v0.4s}, [%2], #16          \n" // r0

        // "prfm   pldl1keep, [%3, #512]       \n"
        // "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

        // "fmla   v16.4s, v8.4s, v0.s[0]      \n"
        // "fmla   v16.4s, v9.4s, v0.s[1]      \n"

        // "subs   %w0, %w0, #1                \n"

        // "fmla   v16.4s, v10.4s, v0.s[2]     \n"
        // "fmla   v16.4s, v11.4s, v0.s[3]     \n"

        // "bne    0b                          \n"

        // "st1    {v16.4s}, [%1], #16         \n"

        // : "=r"(nn),         // %0
        // "=r"(output0_tm), // %1
        // "=r"(r0),         // %2
        // "=r"(k0)          // %3
        // : "0"(nn),
        // "1"(output0_tm),
        // "2"(r0),
        // "3"(k0)
        // : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v16");

        fprintf(stderr,"remain = %d , outch = %d \n",remain_outch_start,outch);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);
            fprintf(stderr, "sum start = %f %f %f %f %f %f %f %f\n", output0_tm[0],output0_tm[1], output0_tm[2],output0_tm[3], output0_tm[4],output0_tm[5], output0_tm[6],output0_tm[7]);


            const Mat kernel0_tm = kernel_tm.channel(p / 2 + p % 2);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i+1 < tiles; i+=2)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                    const float* k01 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0
                    __m256 _sum0 = _mm256_set1_ps(0.f);
                    __m256 _sum1 = _mm256_set1_ps(0.f);

                    for (;nn>0;nn--){

                        __m256 _k01 = _mm256_loadu_ps(k01);
                        __m256 _r00 = _mm256_broadcast_ss(r0);
                        __m256 _r01 = _mm256_broadcast_ss(r0+8);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+1);
                        _r01 = _mm256_broadcast_ss(r0+9);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+2);
                        _r01 = _mm256_broadcast_ss(r0+10);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+3);
                        _r01 = _mm256_broadcast_ss(r0+11);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+4);
                        _r01 = _mm256_broadcast_ss(r0+12);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+5);
                        _r01 = _mm256_broadcast_ss(r0+13);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+6);
                        _r01 = _mm256_broadcast_ss(r0+14);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        _k01 = _mm256_loadu_ps(k01);
                        _r00 = _mm256_broadcast_ss(r0+7);
                        _r01 = _mm256_broadcast_ss(r0+15);
                        _sum0 = _mm256_fmadd_ps(_k01, _r00,_sum0);
                        _sum1 = _mm256_fmadd_ps(_k01, _r01,_sum1);

                        
                        k01 += 64;
                        r0 += 16;

                    }
                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output0_tm+8, _sum1);
                    if (r < 5){
                        fprintf(stderr, "2 sum %d r0 = %f = %f %f %f %f %f %f %f %f\n",r,r0[-16], output0_tm[0],output0_tm[1], output0_tm[2],output0_tm[3],  output0_tm[4],output0_tm[5], output0_tm[6],output0_tm[7]);
                    }
                    output0_tm += 16;
                   
                }

                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

                    const float* k01 = kernel0_tm.row(r);

                    int nn = inch; // inch always > 0
                    __m256 _sum0 = _mm256_set1_ps(0.f);

                    for (;nn>0;nn--){

                        __m256 _k01 = _mm256_loadu_ps(k01);
                        __m256 _r0 = _mm256_set1_ps(r0[0]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);

                        _k01 = _mm256_loadu_ps(k01+8);
                        _r0 = _mm256_set1_ps(r0[1]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0 );

                        _k01 = _mm256_loadu_ps(k01+16);
                        _r0 = _mm256_set1_ps(r0[2]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0 );

                        _k01 = _mm256_loadu_ps(k01+24);
                        _r0 = _mm256_set1_ps(r0[3]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);
                        _k01 = _mm256_loadu_ps(k01+32);
                        _r0 = _mm256_set1_ps(r0[4]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);
                        _k01 = _mm256_loadu_ps(k01+40);
                        _r0 = _mm256_set1_ps(r0[5]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);

                        _k01 = _mm256_loadu_ps(k01+48);
                        _r0 = _mm256_set1_ps(r0[6]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);

                        _k01 = _mm256_loadu_ps(k01+56);
                        _r0 = _mm256_set1_ps(r0[7]);
                        _sum0 = _mm256_fmadd_ps(_k01, _r0,_sum0);


                        k01 += 64;
                        r0 += 8;

                    }
                    _mm256_storeu_ps(output0_tm, _sum0);
                    
                    output0_tm += 8;
                   
                }

            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot
    fprintf(stderr, "top_blob_tm 0 = %f %f %f %f %f %f %f %f\n", top_blob_tm[0],top_blob_tm[1],top_blob_tm[2],top_blob_tm[3],top_blob_tm[4],top_blob_tm[5],top_blob_tm[6],top_blob_tm[7]);
    fprintf(stderr, "top_blob_tm 1 = %f %f %f %f %f %f %f %f\n", top_blob_tm.row(1)[0],top_blob_tm.row(1)[1],top_blob_tm.row(1)[2],top_blob_tm.row(1)[3],top_blob_tm.row(1)[4],top_blob_tm.row(1)[5],top_blob_tm.row(1)[6],top_blob_tm.row(1)[7]);
    fprintf(stderr, "top_blob_tm 5 = %f %f %f %f %f %f %f %f\n", top_blob_tm.row(5)[0],top_blob_tm.row(5)[1],top_blob_tm.row(5)[2],top_blob_tm.row(5)[3],top_blob_tm.row(5)[4],top_blob_tm.row(5)[5],top_blob_tm.row(5)[6],top_blob_tm.row(5)[7]);

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        //         const float otm[6][8] = {
        //             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        //         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            //             const float bias0 = bias ? bias[p] : 0.f;
            __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);

            float tmp[6][8][8];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 16;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 24;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 32;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 40;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 48;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 56;

                    float* output0 = out0.row(i * 6) + (j * 6) * 8;

                    // TODO neon optimize
                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _out0tm0 = _mm256_loadu_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_loadu_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_loadu_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_loadu_ps(output0_tm_3);
                        __m256 _out0tm4 = _mm256_loadu_ps(output0_tm_4);
                        __m256 _out0tm5 = _mm256_loadu_ps(output0_tm_5);
                        __m256 _out0tm6 = _mm256_loadu_ps(output0_tm_6);
                        __m256 _out0tm7 = _mm256_loadu_ps(output0_tm_7);

                        __m256 _tmp024a = vaddq_f32(_out0tm1, _out0tm2);
                        __m256 _tmp135a = vsubq_f32(_out0tm1, _out0tm2);

                        //                         float tmp024a = output0_tm[1] + output0_tm[2];
                        //                         float tmp135a = output0_tm[1] - output0_tm[2];

                        __m256 _tmp024b = vaddq_f32(_out0tm3, _out0tm4);
                        __m256 _tmp135b = vsubq_f32(_out0tm3, _out0tm4);

                        //                         float tmp024b = output0_tm[3] + output0_tm[4];
                        //                         float tmp135b = output0_tm[3] - output0_tm[4];

                        __m256 _tmp024c = vaddq_f32(_out0tm5, _out0tm6);
                        __m256 _tmp135c = vsubq_f32(_out0tm5, _out0tm6);

                        //                         float tmp024c = output0_tm[5] + output0_tm[6];
                        //                         float tmp135c = output0_tm[5] - output0_tm[6];

                        __m256 _tmp0m = vaddq_f32(vaddq_f32(_out0tm0, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f));
                        __m256 _tmp2m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f);
                        __m256 _tmp4m = vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f);
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[4][m], _tmp4m);

                        //                         tmp[0][m] = output0_tm[0] + tmp024a + tmp024b + tmp024c * 32;
                        //                         tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                        //                         tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        __m256 _tmp1m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f);
                        __m256 _tmp3m = vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f);
                        __m256 _tmp5m = vaddq_f32(vaddq_f32(_out0tm7, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f));
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
                        _mm256_storeu_ps(tmp[5][m], _tmp5m);

                        //                         tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        //                         tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                        //                         tmp[5][m] = output0_tm[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0_tm_0 += tiles * 64;
                        output0_tm_1 += tiles * 64;
                        output0_tm_2 += tiles * 64;
                        output0_tm_3 += tiles * 64;
                        output0_tm_4 += tiles * 64;
                        output0_tm_5 += tiles * 64;
                        output0_tm_6 += tiles * 64;
                        output0_tm_7 += tiles * 64;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_loadu_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_loadu_ps(tmp[m][7]);

                        __m256 _tmp024a = vaddq_f32(_tmp01, _tmp02);
                        __m256 _tmp135a = vsubq_f32(_tmp01, _tmp02);

                        //                         float tmp024a = tmp0[1] + tmp0[2];
                        //                         float tmp135a = tmp0[1] - tmp0[2];

                        __m256 _tmp024b = vaddq_f32(_tmp03, _tmp04);
                        __m256 _tmp135b = vsubq_f32(_tmp03, _tmp04);

                        //                         float tmp024b = tmp0[3] + tmp0[4];
                        //                         float tmp135b = tmp0[3] - tmp0[4];

                        __m256 _tmp024c = vaddq_f32(_tmp05, _tmp06);
                        __m256 _tmp135c = vsubq_f32(_tmp05, _tmp06);

                        //                         float tmp024c = tmp0[5] + tmp0[6];
                        //                         float tmp135c = tmp0[5] - tmp0[6];

                        __m256 _out00 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp00, _tmp024a), vmlaq_n_f32(_tmp024b, _tmp024c, 32.f)));
                        __m256 _out02 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 4.f), _tmp024c, 8.f));
                        __m256 _out04 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp024a, _tmp024b, 16.f), _tmp024c, 2.f));
                        _mm256_storeu_ps(output0, _out00);
                        _mm256_storeu_ps(output0 + 16, _out02);
                        _mm256_storeu_ps(output0 + 32, _out04);

                        //                         output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                        //                         output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                        //                         output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                        __m256 _out01 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 2.f), _tmp135c, 16.f));
                        __m256 _out03 = vaddq_f32(_bias0, vmlaq_n_f32(vmlaq_n_f32(_tmp135a, _tmp135b, 8.f), _tmp135c, 4.f));
                        __m256 _out05 = vaddq_f32(_bias0, vaddq_f32(vaddq_f32(_tmp07, _tmp135a), vmlaq_n_f32(_tmp135c, _tmp135b, 32.f)));
                        _mm256_storeu_ps(output0 + 8, _out01);
                        _mm256_storeu_ps(output0 + 24, _out03);
                        _mm256_storeu_ps(output0 + 40, _out05);

                        //                         output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                        //                         output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                        //                         output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                        output0 += outw * 8;
                    }
                }
            }
        }
    }
    fprintf(stderr, "top_blob_bordered = %f %f %f %f\n", top_blob_bordered[0],top_blob_bordered[1],top_blob_bordered[2],top_blob_bordered[3]);
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

