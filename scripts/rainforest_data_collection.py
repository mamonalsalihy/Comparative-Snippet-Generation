import argparse
import collections
import os
import concurrent.futures

import requests as requests
from tqdm import tqdm


class Rainforest:
    """
    sort_by most_helpful --> top_rated on amazon.com
    reviewer_type verified_purchase --> negligible number of reviews filtered.
    review_stars all_positive, all_critical is the same as all_stars (it's a combination of the two)
    Data collection methodology is mostly optimal over all review api parameters, could use search_term for future use
    Reviews have a power law distribution as determined by social importance. Reviews in pages after 1, probably aren't
    socially important.
    """
    global rainforest_url
    rainforest_url = "https://api-dev.braininc.net/be/shopping/rainforest/request"

    def __init__(self, token):
        self.token = token
        self.dev_headers = {
            'Authorization': 'token ' + self.token,
            'Content-Type': 'application/json'
        }

    def get_reviews(self, asins, max_page, save=True, file_path_pattern=None):
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self.review_worker, asin, max_page, save, file_path_pattern) for asin in asins}
            data = [future.result() for future in concurrent.futures.as_completed(futures)]

    def review_worker(self, asin, max_page, save, file_path_pattern):
        product_reviews = collections.defaultdict(list)
        params = {
            'type': 'reviews',
            'amazon_domain': 'amazon.com',
            'asin': asin,
            'sort_by': 'most_helpful',
            'page': 1,
            'max_page': max_page
        }
        response = requests.get(rainforest_url, params=params, headers=self.dev_headers)
        if response.status_code != 200:
            return
        response = response.json()
        print("Number of reviews: {}".format(len(response.get('reviews', []))))
        all_reviews = []
        for res in response.get('reviews', []):
            if res:
                text = res.get('body', "")
                if text != '':
                    all_reviews.append(text)
                else:
                    continue
            else:
                continue
        if save:
            if not os.path.isdir("/".join(file_path_pattern.split("/")[:-1])):
                os.mkdir("/".join(file_path_pattern.split("/")[:-1]))
            self.save_reviews(all_reviews, file_path_pattern.format(asin))
        else:
            product_reviews[asin].append(all_reviews)
        return product_reviews

    def get_products(self, search_term):
        """
        Could later change this to get_asins
        :param search_term:
        :return:
        """
        params = {
            'amazon_domain': 'amazon.com',
            'type': 'search',
            'search_term': search_term
        }
        results = requests.get(rainforest_url, params=params, headers=self.dev_headers).json()['search_results']
        asins = [product['asin'] for product in results]
        return asins

    @staticmethod
    def save_reviews(reviews, file_path):
        with open(file_path, 'w+', encoding='utf8') as fd:
            fd.write("\n\neos-eos\n\n".join(reviews))


def main():
    parser = argparse.ArgumentParser()
    rainforest_parser = parser.add_argument_group('Rainforest')
    rainforest_parser.add_argument('--rainforest_token', type=str, default=None)
    rainforest_parser.add_argument('--search_term', type=str, default=None)
    rainforest_parser.add_argument('--max_page', type=int, default=5)
    rainforest_parser.add_argument('--file_path', type=str, default='../review_data/txt/review_{}.txt')

    args = parser.parse_args()

    rainforest = Rainforest(args.rainforest_token)

    asins = rainforest.get_products(args.search_term)
    product_reviews = rainforest.get_reviews(asins, args.max_page, file_path_pattern=args.file_path)


if __name__ == '__main__':
    main()
