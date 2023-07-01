# Stock movement clusters
:::

::: {.cell .markdown id="YncNbcDYy_1t"}
## Introduction

In this project, we\'ll cluster companies using their daily stock price
movements (i.e. the dollar difference between the closing and opening
prices for each trading day). The NumPy array shows movements of daily
price movements from 2010 to 2015 (obtained from Yahoo! Finance), where
each row corresponds to a company, and each column corresponds to a
trading day.
:::

::: {.cell .markdown}
![stock](vertopal_5b32d6b056aa442ebe24239fe1209e44/1b97704a15df85f904f8703be7762500286c33b9.jpg)
:::

::: {.cell .markdown id="Nt_b0G981yih"}
## Modules
:::

::: {.cell .code execution_count="34" id="rFyaBFdyy1Vm"}
``` python
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import Normalizer
from sklearn.preprocessing import Normalizer
# Import normalize
from sklearn.preprocessing import normalize

from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt

# Import TSNE
from sklearn.manifold import TSNE

```
:::

::: {.cell .markdown id="lVG_Rs671xAL"}
## Dataset
:::

::: {.cell .code execution_count="14" id="HqHWSiPjy4UP"}
``` python
stock_df = pd.read_csv('company-stock-movements-2010-2015-incl.csv')

stock_df.rename(columns={'Unnamed: 0':'companies'}, inplace=True)

X_stock_df = stock_df.drop(['companies'], axis=1)

movements = X_stock_df.to_numpy()
```
:::

::: {.cell .markdown id="EltCR4R4134g"}
## Fit Pipeline
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":182}" id="MjruJC3f2ogR" outputId="5290aaf9-447e-4047-e453-2ab077d4c3bc"}
``` python
# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
:::

::: {.output .execute_result execution_count="20"}
```{=html}
<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;normalizer&#x27;, Normalizer()),
                (&#x27;kmeans&#x27;, KMeans(n_clusters=10))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;normalizer&#x27;, Normalizer()),
                (&#x27;kmeans&#x27;, KMeans(n_clusters=10))])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">Normalizer</label><div class="sk-toggleable__content"><pre>Normalizer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=10)</pre></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="aQE3AAPX24ND"}
## Predict
:::

::: {.cell .code execution_count="21" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="oQLvsetny8jT" outputId="c223a3ac-1b38-4591-9cba-beef2732b2f5"}
``` python
# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
companies = stock_df['companies']
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
```

::: {.output .stream .stdout}
        labels                           companies
    0        0                               Apple
    34       1                          Mitsubishi
    45       1                                Sony
    21       1                               Honda
    48       1                              Toyota
    7        1                               Canon
    15       1                                Ford
    8        2                         Caterpillar
    10       2                      ConocoPhillips
    12       2                             Chevron
    13       2                   DuPont de Nemours
    57       2                               Exxon
    32       2                                  3M
    53       2                       Valero Energy
    44       2                        Schlumberger
    39       2                              Pfizer
    5        3                     Bank of America
    1        3                                 AIG
    26       3                      JPMorgan Chase
    3        3                    American express
    18       3                       Goldman Sachs
    16       3                   General Electrics
    55       3                         Wells Fargo
    37       4                            Novartis
    47       4                            Symantec
    49       4                               Total
    35       4                            Navistar
    42       4                   Royal Dutch Shell
    43       4                                 SAP
    33       4                           Microsoft
    31       4                           McDonalds
    23       4                                 IBM
    58       4                               Xerox
    50       4  Taiwan Semiconductor Manufacturing
    51       4                   Texas instruments
    52       4                            Unilever
    24       4                               Intel
    20       4                          Home Depot
    19       4                     GlaxoSmithKline
    11       4                               Cisco
    54       4                            Walgreen
    6        4            British American Tobacco
    30       4                          MasterCard
    46       4                      Sanofi-Aventis
    41       5                       Philip Morris
    22       6                                  HP
    14       6                                Dell
    29       7                     Lookheed Martin
    36       7                    Northrop Grumman
    4        7                              Boeing
    38       8                               Pepsi
    28       8                           Coca Cola
    27       8                      Kimberly-Clark
    25       8                   Johnson & Johnson
    9        8                   Colgate-Palmolive
    56       8                            Wal-Mart
    40       8                      Procter Gamble
    17       9                     Google/Alphabet
    2        9                              Amazon
    59       9                               Yahoo
:::
:::

::: {.cell .markdown id="OmHERpV37UQY"}
## Hierarchies of stocks
:::

::: {.cell .code execution_count="32" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0HmuqOEm71ip" outputId="25fb8fd8-c9e9-491c-b555-243e08e136da"}
``` python
movements_companies = movements[:60]
len(companies)
```

::: {.output .execute_result execution_count="32"}
    60
:::
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":417}" id="XLSEuzxs2O6l" outputId="2164fc5c-d909-4d52-d65e-6a8808ec04c3"}
``` python
# Normalize the movements: normalized_movements
normalized_movements = normalize(movements_companies)

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()

```

::: {.output .display_data}
![](vertopal_5b32d6b056aa442ebe24239fe1209e44/7c6fff30fead50da4157937da575c25665e384ca.png)
:::
:::

::: {.cell .code id="Xcr79cuH7qRg"}
``` python
```
:::
