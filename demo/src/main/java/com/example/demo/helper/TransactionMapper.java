package com.example.demo.helper;

import com.example.demo.model.FeatureDefinition;
import com.example.demo.model.FeatureType;
import com.example.demo.model.TransactionDTO;
import com.example.demo.model.SyntheticTransaction;
import org.springframework.data.util.Pair;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static com.example.demo.helper.FeatureMappers.*;

@Component
public class TransactionMapper {

    private static final int GAN_FEATURE_DIM = 10;
    private static final int GAN_PRODUCT_DIM = 50;

    // GAN feature output range
    static final float FEATURE_RANGE_MIN = -1.0f;
    static final float FEATURE_RANGE_MAX = 1.0f;
    static final float FEATURE_RANGE_SIZE = FEATURE_RANGE_MAX - FEATURE_RANGE_MIN; // 2.0f


    private String mapValueToCategory(float value, List<String> categories, FeatureType featureType) {

        switch (featureType) {
            case LABEL -> {
                return getForLabels(value, categories);
            }
            case DATE -> {
                return getForDate(value, categories);
            }
            case PHONE_NUMBER -> {
                return getForPhoneNumber(value, categories);
            }
            case NUMBER -> {
                return getForNumber(value, categories);
            }
            case CARD_NUMBER -> {
                return getForCardNumber(value, categories);
            }
        }
        throw new IllegalArgumentException("Unsupported feature type: " + featureType);

    }



    public List<TransactionDTO> mapToDTO(
            List<SyntheticTransaction> syntheticTransactions,
            List<FeatureDefinition> featureDefinitions,
            List<String> productNames) {

        // --- Basic Validation and Warnings ---
        if (featureDefinitions.size() > GAN_FEATURE_DIM) {
            System.err.println("Warning: Feature definitions list size (" + featureDefinitions.size() +
                    ") is greater than the total generated features (" + GAN_FEATURE_DIM +
                    "). Only the first " + GAN_FEATURE_DIM + " will be considered.");
        }
        if (productNames.size() < GAN_PRODUCT_DIM) {
            System.err.println("Warning: Product name list size (" + productNames.size() +
                    ") is less than the total GAN output size (" + GAN_PRODUCT_DIM +
                    "). Generated IDs greater than " + (productNames.size() - 1) + " will be marked invalid.");
        }

        return syntheticTransactions.stream()
                .map(st -> convertToDto(st, featureDefinitions, productNames))
                .collect(Collectors.toList());
    }

    private TransactionDTO convertToDto(
            SyntheticTransaction st,
            List<FeatureDefinition> featureDefinitions,
            List<String> productNames) {

        TransactionDTO dto = new TransactionDTO();
        dto.setQuantity(st.getQuantity());

        // 1. Map Product ID (Numeric Index -> String Name)
        int productIdIndex = st.getProductId();

        int divider = (int) Math.ceil( (double) GAN_PRODUCT_DIM /productNames.size());
        productIdIndex = productIdIndex / divider;

        // ONLY map if the generated ID is within the bounds of the provided names list.
        if (productIdIndex >= 0 && productIdIndex < productNames.size()) {
            dto.setProductName(productNames.get(productIdIndex));
        } else {
            // Case where the GAN generated an ID we don't have a name for (e.g., ID 45, but list size is 40).
            dto.setProductName("INVALID_PRODUCT_ID_" + productIdIndex + "_OOB");
        }

        // 2. Map Conditions (List<Float> -> Map<String, String> of Categories)
        Map<String, String> featureMap = new HashMap<>();
        List<Float> conditions = st.getConditions();

        // Iterate through the generated conditions (up to GAN_FEATURE_DIM)
        for (int i = 0; i < conditions.size() && i < GAN_FEATURE_DIM; i++) {

            // Only proceed if a definition exists for this index (i < featureDefinitions.size())
            if (i < featureDefinitions.size()) {
                float continuousValue = conditions.get(i);

                FeatureDefinition definition = featureDefinitions.get(i);
                String featureName = definition.getName();
                List<String> categories = definition.getValues();

                String categoryName = mapValueToCategory(continuousValue, categories,definition.getType());
                featureMap.put(featureName, categoryName);
            } else {
                break;
            }
        }

        dto.setFeatureMap(featureMap);

        return dto;
    }
}