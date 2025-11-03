package com.example.demo.model;

import lombok.Data;
import java.util.Map;

@Data
public class TransactionDTO {
    private int quantity;
    private String productName;

    // 🔥 CRITICAL CHANGE: The map value type is now String (the category name)
    private Map<String, String> featureMap;
}