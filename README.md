# LLM-empowered KD


**metadata.json:** items content information (raw)

- **mapped_item_content.json:** same as above, with first mapping on items
- **used_item.json:** same as above, with k-core selection (k=10)

**review_warm.json:** Interaction between users and items (raw)

**asin_mapping.json:** item mapping (starting with 100000000)

**reviewerID_mapping.json:** user mapping (starting with 1)

**total.csv:** all interactions (user and item after first mapping)

**BERT_encodings.json:** get item content embedding from **used_item.json** (dim=768)

## Process Code

1. **get_item:**

   - input: metadata.json
   - output: mapped_item_content.json & asin_mapping.json
2. **get_total:**

   - input: review_warm.json
   - output: reviewerID_mapping.json & total.csv
3. **get_used_item:**

   - input: total.csv & mapped_item_content.json
   - output: used_item.json
4. **BERT_encode:**

   - input: used_item.json
   - output: BERT_encodings.json
5. **get_dist_shift:**

   - input: total.csv & used_item.json
   - output: history.csv & genre_fiction.csv
6. **separate_genre_fiction:** k-core selection (k=10)

   - input: genre_fiction.csv
   - output: warm_dense.csv & gf_train.csv & gf_val.csv & gf_test.csv
7. **separate_history:** k-core selection (k=5)

   - input: history.csv
   - output: cold_dense.csv & history_val.csv & history_test.csv
8. **final_map:**

   - input: warm_dense.csv & cold_dense.csv
   - output: user_mapping.json & item_mapping.json
9. **prepare_csv:**

   - input: gf_train.csv & gf_val.csv & gf_test.csv & history_val.csv & history_test.csv & *user_mapping.json & item_mapping.json*
   - output: warm_train.csv & warm_val.csv & warm_test.csv & cold_val.csv & cold_test.csv
10. **prepare_BERT:**

- input: BERT_encodings.json & *user_mapping.json & item_mapping.json*
- output: item_content_emb.npy

11. **prepare_neighbourhood:**

- input: warm_train.csv & warm_val.csv & warm_test.csv & cold_val.csv & cold_test.csv
- output: item_content_emb.npy
