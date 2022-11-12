package com.example.yolov5;

public record Detection (String label, float[] bbox, float confidence) {

}
