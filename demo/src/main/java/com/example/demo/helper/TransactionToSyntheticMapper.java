package com.example.demo.helper;

import com.example.demo.model.TransactionDTO;
import com.example.demo.model.SyntheticTransaction;
import org.springframework.data.util.Pair;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
public class TransactionToSyntheticMapper {

    private static final int GAN_FEATURE_DIM = 10;
    private static final int GAN_PRODUCT_DIM = 50;

    // GAN feature output range
    private static final float FEATURE_RANGE_MIN = -1.0f;
    private static final float FEATURE_RANGE_MAX = 1.0f;
    private static final float FEATURE_RANGE_SIZE = FEATURE_RANGE_MAX - FEATURE_RANGE_MIN; // 2.0f

    private float mapCategoryToMidpoint(String categoryName, List<String> categories) {
        if (categories == null || categories.isEmpty()) {
            return 0.0f;
        }

        int numBins = categories.size();

        // Explicit float calculation
        float binWidth = FEATURE_RANGE_SIZE / (float) numBins;
        int binIndex = categories.indexOf(categoryName);

        if (binIndex == -1) {
            return 0.0f; // Category name not found
        }

        // 1. Calculate the raw midpoint value
        float binStart = FEATURE_RANGE_MIN + (binIndex * binWidth);
        float midpoint = binStart + (binWidth / 2.0f);

        // 2. ENFORCE 7 DECIMAL PLACES
        // Multiply by 10 million (10^7), round to the nearest whole number, and divide back.
        final float scaleFactor = 10000000.0f; // 10^7

        // Note: Math.round takes a float and returns an int/long.
        return Math.round(midpoint * scaleFactor) / scaleFactor;
    }

    public List<SyntheticTransaction> mapToSynthetic(
            List<TransactionDTO> transactionDTOs,
            List<Pair<String, List<String>>> featureDefinitions,
            List<String> productNames) {

        return transactionDTOs.stream()
                .map(dto -> convertToSynthetic(dto, featureDefinitions, productNames))
                .collect(Collectors.toList());
    }

    private SyntheticTransaction convertToSynthetic(
            TransactionDTO dto,
            List<Pair<String, List<String>>> featureDefinitions,
            List<String> productNames) {

        SyntheticTransaction st = new SyntheticTransaction();
        st.setQuantity(dto.getQuantity());

        // 1. Map Product Name to Numeric ID (Inverse of the division logic)
        String productName = dto.getProductName();
        int providedIndex = productNames.indexOf(productName);

        if (providedIndex != -1) {
            // Reversing the division: ID = Index * Divider
            int divider = (int) Math.ceil( (double) GAN_PRODUCT_DIM / productNames.size());
            st.setProductId(providedIndex * divider);
        } else {
            st.setProductId(0);
        }

        // 2. Map Categorical Features back to Continuous Floats
        List<Float> conditions = new ArrayList<>(GAN_FEATURE_DIM);
        Map<String, String> dtoFeatureMap = dto.getFeatureMap();

        for (int i = 0; i < GAN_FEATURE_DIM; i++) {
            float continuousValue;

            if (i < featureDefinitions.size()) {
                // Case A: Feature was categorized. Find midpoint.
                Pair<String, List<String>> definition = featureDefinitions.get(i);
                String featureName = definition.getFirst();
                List<String> categories = definition.getSecond();

                String categoryName = dtoFeatureMap.get(featureName);

                if (categoryName != null) {
                    continuousValue = mapCategoryToMidpoint(categoryName, categories);
                } else {
                    // Feature name present in definitions but missing from DTO map, default to 0.0f
                    continuousValue = 0.0f;
                }
            } else {
                continuousValue = 0.0f;
            }
            conditions.add(continuousValue);
        }

        st.setConditions(conditions);

        return st;
    }
}