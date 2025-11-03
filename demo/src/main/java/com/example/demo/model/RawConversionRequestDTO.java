package com.example.demo.model;

import org.springframework.data.util.Pair;
import lombok.Data;

import java.util.List;

/**
 * DTO for the raw conversion POST endpoint. Contains the categorized data
 * and the necessary metadata (definitions) for the reverse mapping.
 */
@Data
public class RawConversionRequestDTO {

    // The categorized data provided by the client
    private String csvFile;

    // The definitions used to discretize the continuous features (Name -> Category List)
    private List<Pair<String, List<String>>> featureDefinitions;

    // The names of all possible products (used for re-indexing)
    private List<String> productNames;
}