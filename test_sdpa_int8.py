import torch
import time
def func(q, k, v):
    output = torch._scaled_dot_product_int8(
        q,
        k,
        v,
        attn_mask=None,
        scale=0.125,  # scale
        dropout_p=0.0,  # dropout
        is_causal=False,  # is_causal
        q_zp=0,
        q_scale=1,
        k_zp=0,
        k_scale=1,
        v_zp=0,
        v_scale=1,
        a_zp=0,
        a_scale=1,
        o_zp=0,
        o_scale=1,
    )
    return output
torch.manual_seed(2025)

# shapes = [(1, 12, 197, 64), (120, 12, 197, 64), (1, 16, 384, 64), (120, 16, 384, 64), (16, 32, 1024, 128), (1, 16, 1024, 64)]
shapes = [(120, 16, 384, 64),]
for shape in shapes:
    print("-----------shape------------", shape)
    q = torch.randn(shape, dtype=torch.float, device='cpu') * 10
    k = torch.randn(shape, dtype=torch.float, device='cpu') * 10
    v = torch.randn(shape, dtype=torch.float, device='cpu') * 10
    q = q.to(torch.uint8)
    k = k.to(torch.uint8)
    v = v.to(torch.uint8)

    warmup = 3000
    itern = 3000


    for _ in range(warmup):
        ref_res = func(q, k, v)

    start = time.time()
    for _ in range(itern):
        ref_res = func(q, k, v)
    end = time.time()

    print("Iter time: ", (end - start) * 1000 / itern)
# ref_res = func(q, k, v)
# print(ref_res)
# compiled_func = torch.compile(func, backend="inductor")

# res = compiled_func(q, k, v)
# print(torch.testing.assert_close(ref_res, res, atol=0.01, rtol=0.01), flush=True)


'''
graph api

(16, 64, 120, 64)
single core
Iter time:  23.498398303985596
8 cores
Iter time:  5.015988111495972

(4, 64, 256, 128)
single core
Iter time:  20.735906839370728
8 cores
Iter time:  3.758122444152832

40 cores:

-----------shape------------ (1, 12, 197, 64)
Iter time:  0.14752785364786783
-----------shape------------ (120, 12, 197, 64)
Iter time:  2.591773271560669
-----------shape------------ (1, 16, 384, 64)
Iter time:  0.17553067207336426
-----------shape------------ (120, 16, 384, 64)
Iter time:  8.358789920806885
-----------shape------------ (16, 32, 1024, 128)
Iter time:  21.31683087348938
-----------shape------------ (1, 16, 1024, 64)
Iter time:  0.8591973781585693
'''

'''
brgemm api

(16, 64, 120, 64)
single core
Iter time:  24.114911317825317

8 core
Iter time:  3.428818702697754

(4, 64, 256, 128)
single core
Iter time:  22.371724367141724
8 cores
Iter time:  3.2345640659332275

40 cores:

-----------shape------------ (1, 12, 197, 64)
Iter time:  0.12720727920532227
-----------shape------------ (120, 12, 197, 64)
Iter time:  3.126844644546509
-----------shape------------ (1, 16, 384, 64)
Iter time:  0.17683076858520508
-----------shape------------ (120, 16, 384, 64)
Iter time:  8.208382368087769
-----------shape------------ (16, 32, 1024, 128)
Iter time:  22.165619134902954
-----------shape------------ (1, 16, 1024, 64)
Iter time:  0.6715164184570312

'''