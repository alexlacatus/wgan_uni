package com.example.demo.model;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
@AllArgsConstructor
public class FeatureDefinition {
    private String name;
    private List<String> values;
    private FeatureType type;
}
