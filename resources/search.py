from flask import request
from flask.ext import restful
from flask.ext.restful import reqparse
from engine.matcher import process_image
import werkzeug


# Constants
DEFAULT_MAX_IMAGES_FOUND = 5

# Argument parser
search_parser = reqparse.RequestParser()

search_parser.add_argument(
    'min_score',
    type=float, location='values',
    default=0.0, help='The minimum score, between 0.0 and 1.0',
)
search_parser.add_argument(
    'max_results',
    type=int, location='values',
    default=DEFAULT_MAX_IMAGES_FOUND, help='The maximum number of results',
)
search_parser.add_argument(
    'image',
    type=werkzeug.datastructures.FileStorage, location='files',
    required=True,
    help='The image to search for',
)


class SearchAPI(restful.Resource):
    def post(self):
        args = search_parser.parse_args()

        results, status_code = process_image(args.image,
                                             args.max_results,
                                             args.min_score)
        if status_code == 200:
            return {"count": len(results), "results": results}
        else:
            restful.abort(status_code, message=results)
