import pandas as pd

x = pd.DataFrame({'author_1': [1, 2, 3, 5, 6, 2],
                   'abstract': [3, 4, 3, 2, 1, 3],
                   'author_2': [1, 2, 3, 1, 1, 5],
                   'title': [1, 2, 4, 5, 9, 3],
                   'graph': [1, 2, 4, 5, 9, 4]})

y = pd.DataFrame({'target': [0, 1, 1, 0, 1, 0]})

test_x = pd.DataFrame({'author_1': [1, 2, 3],
                        'abstract': [3, 4, 3],
                        'author_2': [1, 2, 3],
                        'title': [1, 2, 4],
                        'graph': [1, 2, 4]})
