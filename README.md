# Cosine distance evaluation

Đánh giá precisions và recall theo các threshold

## News

### July 26, 2022

* Cập nhật Phiên bản 0.1.0

## Yêu cầu

Data format theo định dạng sau:

```
data_dir
└───id1
│   │   0.jpg
│   │   1.jpg
│   │   *.jpg
│
└───id2
...
```

Chuyển đổi định dạng sử dụng: 
`python tools/reformat.py -c [config_path]`

## Tutorial

1. Cấu hình config file
2. Viết run.py theo template, cấu hình prepocess, augment, postprocess...
4. `python run.py -c [config_path]`