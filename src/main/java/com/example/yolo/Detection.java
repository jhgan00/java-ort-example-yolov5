package com.example.yolo;

public record Detection(String label, float[] bbox, float confidence) {

}
