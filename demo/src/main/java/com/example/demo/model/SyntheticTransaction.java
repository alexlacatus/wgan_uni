package com.example.demo.model;// src/main/java/.../model/SyntheticTransaction.java

import lombok.Data;

import java.util.List;

@Data
public class SyntheticTransaction {
    private int quantity;
    private int productId;
    private List<Float> conditions;
}