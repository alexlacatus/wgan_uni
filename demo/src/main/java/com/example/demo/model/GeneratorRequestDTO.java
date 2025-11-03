package com.example.demo.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.springframework.data.util.Pair;

import java.util.List;

@Data
@AllArgsConstructor
public class GeneratorRequestDTO {
    private List<FeatureDefinition> featureDefinitions;
    private int size;
    private List<String> productNames;
}
