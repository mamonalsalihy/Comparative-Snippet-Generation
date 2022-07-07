

# Details



## Data Collection

Product reviews are collected from Amazon using the following script `rainforest_data_collection.py`.

It contains two different functionalities: 
1. Gets products given a `search_term`
2. Gets product reviews and saves them with the following file format: `review_{asin}.txt` 

The API request for getting products using a `search_term` follow the following format:
 
      import requests
      
      # set up the request parameters
       rainforest_url = "https://api-dev.braininc.net/be/shopping/rainforest/request"

      self.dev_headers = {
            'Authorization': 'token ' + self.token,
            'Content-Type': 'application/json'
      }
      params = {
        'type': 'search',
        'amazon_domain': 'amazon.com'
        'search_term': 'sunscreen'
      }
      
      # make the http GET request to Rainforest API
      api_results = requests.get(rainforest_url, params=params, headers=self.dev_headers)
      
      # print the JSON response from Rainforest API
      print(json.dumps(api_result['search_results'].json()))

The API request for getting product reviews follow the following format:

         import requests
         
         # set up the request parameters
          rainforest_url = "https://api-dev.braininc.net/be/shopping/rainforest/request"
   
         self.dev_headers = {
               'Authorization': 'token ' + self.token,
               'Content-Type': 'application/json'
         }

         params = {
             'type': 'reviews',
             'amazon_domain': 'amazon.com',
             'asin': asin,
             'sort_by': 'most_helpful'
         }
         api_result = requests.get(rainforest_url, params=params, headers=self.dev_headers)
         
         # print the JSON response from Rainforest API
         print(json.dumps(api_result['reviews'].json()))

## Review Segmentation



### Feng Hirst Parser

Changes I made to the original code base are in `Dockerfile` (there might be more steps that I am forgetting)
```
WORKDIR /opt
COPY . /opt/feng-hirst-rst-parser
```

Since the previous step is already completed by cloning the project, proceed with doing the following steps: 
1. Add all review files to `feng-hirst-rst-parser/tmp/txt` 

2. Give run permission to `get_edus.sh` which will run the docker container for each file in `/tmp/txt`

    ```
    chmod +x get_edus.sh
    ./get_edus.sh
    ```
### Notes
1. Some reviews are not in unicode so edu segmentation fails.
2. Suggestion: Create a separate terminal process to run `./get_edus.sh`
3. For each file, we need to create a new docker container and mount the same directory
4. The results will be stored in `./tmp`



### Post-processing segments
Filtering rules are given in: [Comparative_snippet_generation](github.com/WING-NUS/comparative-snippet-generation-dataset)

`review_segmentation.py` accepts `review_{asin}.txt.edus` files and returns `review_segment_{asin}.txt.edus`


## Segment Polarity classification

The classifier uses the following as its training set:

The SPOT dataset contains 197 reviews originating from the Yelp'13 and IMDB collections ([1][2]),
annotated with segment-level polarity labels (positive/neutral/negative). Annotations 
have been gathered on 2 levels of granulatiry:

 - Sentences
 - Elementary Discourse Units (EDUs), i.e. sub-sentence clauses produced by a state-of-the-art
RST parser 

Statistics and details about the dataset's creation can be found in our paper:

> **Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis**,<br/>
> Stefanos Angelidis, Mirella Lapata. 2017. <br/>
> _To appear in Transactions of the Association for Computational Linguistics (TACL)_.<br/>
> [ [arXiv](https://arxiv.org/abs/1711.09645) ]


I decided to convert the `txt` files into `csv` files and do some filtering. There are three sentiment labels in `SPOT`: positive,negative,neutral.
I removed neutral segments and removed some special tokens e.g. `<s>`. The functionality is provided in `convert_txt_csv`. 

We label amazon product review segments with our classifier using the following script:

`segment_polarity_classification.py`

The file will write to `./review_data/{positive,negative}_segments/review_segment_{asin}.txt.edus`
Some files may not have corresponding positive/negative sentiment segments.

Note: need auth_token since model was trained using auto_train from huggingface


## Opinion Summarization

This is the script for data formatting `qt_custom_data.py`. Formatting is used twice.
Once for creating the training set and once for creating the test set.

### Data format conversion

We need to prepare your dataset in the appropriate json
format. Here is how the training set should look like:

```
[
  {
    "entity_id": "...",
    "reviews": [
      {
        "review_id": "...",
        "rating": 3,
        "sentences": [
          "first sentence text",
          "second sentence text", 
          ...
        ]
      },
      ...
    ]
  },
  ...
]
```

All review segments are used for training the quantized transformer.


The output will be stored as `../qt/data/json/qt_train.json`

For positive/negative segments, the output will be stored as `../qt/data/json/qt_{pos/neg}_test.json`


### Opinion Extraction
````
python3 extract.py --summary_data ../data/json/mydata_summ.json --no_eval --sentencepiece ../data/sentencepiece/myspm.model --split_by presplit --model ../models/mydata_run1_20_model.pt --sample_sentences --gpu 0 --run_id mydata_gen_run1
````

Summaries are stored in `../qt/outputs/general_run1` where each file corresponds to a product




## Places to improve the pipeline
### Data collection
   1. Use `most-helpful` product reviews (snippets should be informative about the product)
   2. reviewer_type: `verified_purchase` (reviews should be by real people who have used the product)
   3. `search_term` (could use key words to help inform which subset of the reviews could be most informative/relevant, useful for personalization) 
   4. `max_page` we want to process 100s-1000s of review per product to get better general opinion summaries 
      1. API credits will be used for each page
   5. `show_different_asins` this could be used for getting batched product reviews
   6. File saving first aggregates all reviews for each product. Instead, we could iteratively save reviews to the corresponding file
      ````
      all_reviews = []
      for res in response.get('reviews', []):
         if res:
           text = res.get('body', "")
           if text != '':
              if save:
                self.save_reviews(all_reviews, file_path_pattern.format(asin))
              else:
                continue
           else:
             continue
      ````
      
### Feng-Hirst-RST-Parser

This is the command I use for segmentation (could automate by writing shell script)


### Review segment PostProcessing 

1.Could add more part of speech patterns for accepting valid segments



### Opinion Summarization

1. Finally, you need to train a SentencePiece tokenizer on your data using our
`train-spm.py` script

````
    cd ./src/utils/
    python3 train-spm.py path/to/mydata_train.json myspm
    mv myspm* ../../data/sentencepiece/
````

## Limitations
1. Hyperparameter tuning is necessary to figure out what the most representative general opinion summary is
2. The quantized transformer needs to ingest large sums of data for training. 
3. Unclear what the training data should be to generate a good opinion summary is.
   1. Thoughts on this: Should it aggregate review from every product for the search term
   2. Should it be only reviews from each product
   3. What does this mean for training a quantized transformer?
   4. How many quantized transformers would we have?
   5. Per domain, per search term, per product?
4. (Pre/Post)processing is engineering intensive.
   1. This is an issue only because we need to segment review sentences so it could be classified by sentiment
5. Sentence selection could either be done using human annotators or by automatic methods (neural methods or linguistic pre-processing or both)
6. Need to remove characters that are not utf8
