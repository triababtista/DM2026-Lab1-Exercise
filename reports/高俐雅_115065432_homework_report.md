# Report: Homework (Exploring the Reddit Stock Sentiment Dataset)

**Loading, cleaning and length statistics.**\
I loaded `newdataset/Reddit-stock-sentiment.csv` (text column `text`, label column `label`) with `load_dataset_tool` (`load_dataset_1`): 847 short posts from a retail-investing forum, labelled -1.0 (negative, 315), 0.0 (neutral, 423) or 1.0 (positive, 109). The first 20 rows (`inspect_data_2`) show stock jargon, political and tariff commentary, and crude informal language. `check_missing_tool` (`check_missing_3`) found no missing text. `check_duplicates_tool` first reported 23 duplicates (`check_duplicates_4`), and rerunning with dropping enabled (`check_duplicates_5`) left 813 rows. That is 34 fewer rows than the 23 which was previously reported, so the tool apparently removed all copies of the duplicated posts, as it was intentionally designed to remove all duplicate copies and retain only unique data from the beginning. After cleaning, the classes are 0.0: 391, -1.0: 315 and 1.0: 107, so the data is imbalanced and positive posts are only about 13%. `describe_data_tool` (`describe_data_6`) shows very short, right-skewed posts: mean 148.31 characters, median 80, maximum 2,753. Mean lengths by class are similar (-1.0: 153.74, 0.0: 139.08, 1.0: 166.04). Neutral posts have the widest spread (std 260.46) and the most extreme outliers. I also drew random samples with `sample_data_tool` (`sample_data_7`, `sample_data_8`, `sample_data_9`; the last with random state 1, 150 rows) to read more posts and to learn more domain knowledge.

![Text length by category](../plots/高俐雅_homework_plot_describe_data_6.png)

**Tokenization, document-term matrix and term filtering.**\
`tokenize_tool` (`tokenize_10`, 813 documents) splits text into unigrams, which breaks contractions such as "didn't" into 'didn', '’' and 't' (seen in `inspect_data_11`). `build_dtm_tool` (`build_dtm_12`) gave 4,313 terms and 16,659 non-zero entries out of 3,506,469 cells, a sparsity of 99.5249%. The 20x20 slice in `dtm_heatmap_13` is almost entirely zeros.

![DTM heatmap, first 20 documents x 20 terms](../plots/高俐雅_homework_plot_dtm_heatmap_13.png)

Top term frequencies (`term_frequency_14`) are all stop words: "the" (874), "to" (554), "and" (471), "is" (386), "it" (338). Variance filtering kept all 4,313 terms at thresholds 0.0 (`variance_filter_15`) and 0.0001 (`variance_filter_17`). At 0.005 (`variance_filter_16`) it kept 792 terms and removed 3,521, mostly rare words, but the top kept terms were still stop words ("the", "to", "and", "is", "you"). Variance alone therefore removes rare terms but does not remove stop words. The correlation matrix of the top-20 variance terms (`feature_correlation_22`) is again nearly all stop words, with moderate-to-high pairwise correlations (e.g. "the" and "in" about 0.69) because they all increase as the post gets longer. Data filtering based on variance and term frequency does not provide meaningful insights in this session because English stop words are not removed before the document-term matrix is created, as intentionally designed in the tool.

![Correlation of top-variance terms](../plots/高俐雅_homework_plot_feature_correlation_22.png)

**Class-specific vocabulary (Pearson and Spearman).**\
 For negative posts, Pearson (`pearson_filter_18`) and Spearman (`spearman_filter_19`) agree on the main terms: "he" (0.1762 / 0.1965), "his", "will", "trump" (0.1506 / 0.1539), "fuck" and "economy", with "china" and "trade" appearing only in Pearson's top 10 and "until", "us" and "the" only in Spearman's. For neutral posts (`spearman_filter_20`) the top terms are weak and generic ("year", "always" at 0.1004, "were", "sense", "art", "x200b", "cash"). For positive posts (`spearman_filter_21`) they are "technology" (0.1609), "main", "huge" and "tldr" (all 0.1563), "nice", "potential" and "trend". The tied scores for "main", "huge" and "tldr" suggest they occur in the same few documents, probably the same long write-up. All correlations are low (the highest is about 0.20), so no single word separates a sentiment class well. Still, negative posts lean toward politics and trade, and positive posts toward technology and optimistic wording.

![Spearman, negative class](../plots/高俐雅_homework_plot_turn114.png)
![Spearman, positive class](../plots/高俐雅_homework_plot_turn138.png)

**Pattern mining.**\
I used `mine_patterns_tool` with TF-IDF filtering. Top-k (k=15) for the positive class (`mine_patterns_23`) returned only top 10 single-word patterns, led by "just" (support 15) and "good" (12). For the negative class (`mine_patterns_24`) the top patterns were "trump" (45), "just" (42), "china" (33), "like" (25), "market" and "people" (23 each). MaxFP-Growth with minimum support 10 gave 30 patterns for the negative class (`mine_patterns_25`, shown ones include "bad", "american", "shit", "xi", all support 10-11) and 34 for the neutral class (`mine_patterns_26`: "x200b", "buy", "silver", "price", "stocks", all support 10-12). The MaxFP-Growth algorithm was able to discover hidden patterns that differed from the Top-K algorithm results because it considers more than just pattern support frequency. Negative posts are dominated by Trump, China and the economy, while neutral posts look like ordinary trading chatter.

![MaxFP-Growth patterns, negative class](../plots/高俐雅_homework_plot_turn164.png)

**Dimensionality reduction, binarization and similarity.**\
 PCA (`reduce_dimensions_27`) explains only 26.70% and 5.12% of the variance on its first two components, and its plot packs almost all documents into one tight region with a few far-away outliers. t-SNE (`reduce_dimensions_28`) and UMAP (`reduce_dimensions_29`) spread the points out, but none of the three plots shows clear clusters by sentiment. The classes use largely the same vocabulary, and unigram counts cannot capture tone, negation or sarcasm.

![UMAP of documents by sentiment](../plots/高俐雅_homework_plot_reduce_dimensions_29.png)
![PCA of documents by sentiment](../plots/高俐雅_homework_plot_reduce_dimensions_27.png)
![t-SNE of documents by sentiment](../plots/高俐雅_homework_plot_reduce_dimensions_28.png)

`binarize_labels_tool` (`binarize_labels_30`) produced a 813x3 one-hot label matrix. Cosine similarity (`cosine_similarity_31`, `_32`, `_33`) gave 0.3030 for documents 2 and 8, 0.0 for documents 5 and 7, and 0.2004 for documents 8 and 9. Posts are short, so many pairs share few or no words, even within one class. Overall, the next steps should be stop-word and stray-token removal (e.g. "x200b"), TF-IDF weighting, and handling the class imbalance before applying any algorithm or training any classifier.
