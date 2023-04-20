{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import nltk\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import naive_bayes,metrics\n",
    "from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,accuracy_score\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     /home/codespace/nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download(\"stopwords\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = pd.read_csv('./datasets/reviews.txt',sep = '\\t', names =['Reviews','Comments'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Reviews</th>\n",
       "      <th>Comments</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>The Da Vinci Code book is just awesome.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>this was the first clive cussler i've ever rea...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>i liked the Da Vinci Code a lot.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>i liked the Da Vinci Code a lot.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>I liked the Da Vinci Code but it ultimatly did...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6913</th>\n",
       "      <td>0</td>\n",
       "      <td>Brokeback Mountain was boring.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6914</th>\n",
       "      <td>0</td>\n",
       "      <td>So Brokeback Mountain was really depressing.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6915</th>\n",
       "      <td>0</td>\n",
       "      <td>As I sit here, watching the MTV Movie Awards, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6916</th>\n",
       "      <td>0</td>\n",
       "      <td>Ok brokeback mountain is such a horrible movie.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6917</th>\n",
       "      <td>0</td>\n",
       "      <td>Oh, and Brokeback Mountain was a terrible movie.</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6918 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Reviews                                           Comments\n",
       "0           1            The Da Vinci Code book is just awesome.\n",
       "1           1  this was the first clive cussler i've ever rea...\n",
       "2           1                   i liked the Da Vinci Code a lot.\n",
       "3           1                   i liked the Da Vinci Code a lot.\n",
       "4           1  I liked the Da Vinci Code but it ultimatly did...\n",
       "...       ...                                                ...\n",
       "6913        0                     Brokeback Mountain was boring.\n",
       "6914        0       So Brokeback Mountain was really depressing.\n",
       "6915        0  As I sit here, watching the MTV Movie Awards, ...\n",
       "6916        0    Ok brokeback mountain is such a horrible movie.\n",
       "6917        0   Oh, and Brokeback Mountain was a terrible movie.\n",
       "\n",
       "[6918 rows x 2 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "stopset = stopwords.words('english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = TfidfVectorizer(use_idf = True,lowercase = True, strip_accents='ascii',stop_words=stopset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = vectorizer.fit_transform(dataset.Comments)\n",
    "y = dataset.Reviews\n",
    "pickle.dump(vectorizer, open('tranform.pkl', 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = naive_bayes.MultinomialNB()\n",
    "clf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "97.47109826589595"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test,clf.predict(X_test))*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf = naive_bayes.MultinomialNB()\n",
    "clf.fit(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "cf=confusion_matrix(y_test,clf.predict(X_test))\n",
    "sns.heatmap(cf,annot=True,fmt=\"0.02f\")\n",
    "plt.title(\"Confusion Matrix of Naive Bayes model\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7fc065f00820>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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VOGYO++8U0VFRbriiitkjNGJEyf0wAMPnPGy1KhRo5SWluZ8XVJSQsABAMDC/L6g2BMrV67UhAkT9Nprryk7O1sffvihFi9erGeeeea0Y4KDgxUREeGyAQAA6/LbmZuoqCgFBgaqsLDQpb2wsFCxsbEVjhk7dqz69++v++67T5LUunVrHT58WPfff7+efPJJBQRUq6wGAAB8wG9pwG63q3379srMzHS2ORwOZWZmKikpqcIxR44cOSXABAYGSpKMMb4rFgAAVBt+O3MjSWlpaUpNTVWHDh3UsWNHTZ48WYcPH9bAgQMlSQMGDFC9evWUkZEhSerVq5cmTZqkSy+9VJ06ddLWrVs1duxY9erVyxlyAADA+c2v4aZv377at2+fxo0bp4KCArVt21ZLly51LjLeuXOny5maMWPGyGazacyYMdq9e7cuuOAC9erVS88995y/DgEAAJxjbOY8u55TUlKiyMhIFRcXe3Vx8ZGyE2o5bpkkadPTKQqz+zU3AgBgKZ58f7MCFwAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWArhBgAAWMpZhZtjx455qw4AAACv8DjcOBwOPfPMM6pXr55q1aql7du3S5LGjh2rWbNmeb1AAAAAT3gcbp599lnNmTNHL7zwgux2u7O9VatWmjlzpleLAwAA8JTH4Wbu3LmaPn26+vXrp8DAQGd7mzZt9NNPP3m1OAAAAE95HG52796tiy666JR2h8Oh48ePe6UoAACAyvI43LRs2VKrV68+pf3999/XpZde6pWiAAAAKquGpwPGjRun1NRU7d69Ww6HQx9++KFyc3M1d+5cffrpp76oEQAAwG0en7np3bu3/vnPf2rFihWqWbOmxo0bp82bN+uf//ynrr76al/UCAAA4DaPz9xI0pVXXqnly5d7uxYAAICz5vGZm0aNGuk///nPKe0HDhxQo0aNvFIUAABAZXkcbnbs2KHy8vJT2ktLS7V7926vFAUAAFBZbl+WWrRokfPPy5YtU2RkpPN1eXm5MjMzlZiY6NXiAAAAPOV2uOnTp48kyWazKTU11eW9oKAgJSYmauLEiV4tDgAAwFNuhxuHwyFJatiwob755htFRUX5rCgAAIDK8vhuqby8PF/UAQAA4BWVuhX88OHDWrVqlXbu3KmysjKX94YNG+aVwgAAACrD43CzYcMG9ezZU0eOHNHhw4dVt25dFRUVKSwsTNHR0YQbAADgVx7fCj58+HD16tVLv/76q0JDQ/X111/r3//+t9q3b6+XXnrJFzUCAAC4zeNwk5OTo0cffVQBAQEKDAxUaWmpEhIS9MILL2j06NG+qBEAAMBtHoeboKAgBQT8Niw6Olo7d+6UJEVGRmrXrl3erQ4AAMBDHq+5ufTSS/XNN9+oSZMm6tKli8aNG6eioiK9/fbbatWqlS9qBAAAcJvHZ24mTJiguLg4SdJzzz2nOnXq6MEHH9S+ffv0xhtveL1AAAAAT3h85qZDhw7OP0dHR2vp0qVeLQgAAOBseHzm5nSys7N1/fXXezxu6tSpSkxMVEhIiDp16qR169adsf+BAwc0ePBgxcXFKTg4WE2bNtWSJUsqWzYAALAYj8LNsmXLNGLECI0ePVrbt2+XJP3000/q06ePLrvsMucjGty1YMECpaWlKT09XdnZ2WrTpo1SUlK0d+/eCvuXlZXp6quv1o4dO/T+++8rNzdXM2bMUL169Tz6XAAAYF1uX5aaNWuWBg0apLp16+rXX3/VzJkzNWnSJA0dOlR9+/bVDz/8oBYtWnj04ZMmTdKgQYM0cOBASdK0adO0ePFizZ49WyNHjjyl/+zZs7V//3599dVXCgoKkqQ/fRJ5aWmpSktLna9LSko8qhEAAFQvbp+5mTJliv7+97+rqKhI7733noqKivTaa6/p+++/17Rp0zwONmVlZVq/fr2Sk5P/W0xAgJKTk5WVlVXhmEWLFikpKUmDBw9WTEyMWrVqpQkTJqi8vPy0n5ORkaHIyEjnlpCQ4FGdAACgenE73Gzbtk233nqrJOmmm25SjRo19OKLL6p+/fqV+uCioiKVl5crJibGpT0mJkYFBQUVjtm+fbvef/99lZeXa8mSJRo7dqwmTpyoZ5999rSfM2rUKBUXFzs3fosHAABrc/uy1NGjRxUWFiZJstlsCg4Odt4SXlUcDoeio6M1ffp0BQYGqn379tq9e7defPFFpaenVzgmODhYwcHBVVonAADwH49uBZ85c6Zq1aolSTpx4oTmzJmjqKgolz7uPjgzKipKgYGBKiwsdGkvLCxUbGxshWPi4uIUFBSkwMBAZ1uLFi1UUFCgsrIy2e12Tw4HAABYkNvhpkGDBpoxY4bzdWxsrN5++22XPjabze1wY7fb1b59e2VmZqpPnz6Sfjszk5mZqSFDhlQ45vLLL9e8efPkcDicj4DYsmWL4uLiCDYAAECSB+Fmx44dXv/wtLQ0paamqkOHDurYsaMmT56sw4cPO++eGjBggOrVq6eMjAxJ0oMPPqhXX31VDz/8sIYOHaqff/5ZEyZMcDtQAQAA6/P4F4q9qW/fvtq3b5/GjRungoICtW3bVkuXLnUuMt65c6fzDI0kJSQkaNmyZRo+fLguueQS1atXTw8//LCeeOIJfx0CAAA4x9iMMcbfRVSlkpISRUZGqri4WBEREV7b75GyE2o5bpkkadPTKQqz+zU3AgBgKZ58f3vt8QsAAADnAsINAACwFMINAACwlEqFm23btmnMmDG64447nA+5/Oyzz/Tjjz96tTgAAABPeRxuVq1apdatW+tf//qXPvzwQx06dEiStHHjxtP+SjAAAEBV8TjcjBw5Us8++6yWL1/u8sN53bp109dff+3V4gAAADzlcbj5/vvvdeONN57SHh0draKiIq8UBQAAUFkeh5vatWsrPz//lPYNGzaoXr16XikKAACgsjwON7fffrueeOIJFRQUyGazyeFwaO3atRoxYoQGDBjgixoBAADc5nG4mTBhgpo3b66EhAQdOnRILVu21FVXXaXOnTtrzJgxvqgRAADAbR4/I8But2vGjBkaO3asfvjhBx06dEiXXnqpmjRp4ov6AAAAPOJxuFmzZo2uuOIKNWjQQA0aNPBFTQAAAJXm8WWpbt26qWHDhho9erQ2bdrki5oAAAAqzeNws2fPHj366KNatWqVWrVqpbZt2+rFF1/UL7/84ov6AAAAPOJxuImKitKQIUO0du1abdu2TbfeeqveeustJSYmqlu3br6oEQAAwG1n9eDMhg0bauTIkXr++efVunVrrVq1ylt1AQAAVEqlw83atWv10EMPKS4uTnfeeadatWqlxYsXe7M2AAAAj3l8t9SoUaM0f/587dmzR1dffbWmTJmi3r17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uece5+tPPvlE7dq1U0hIiBo1aqTx48frxIkTZzzOGjVqKDY2VvXq1VNycrJuvfVWLV++3Pl+eXm57r33XjVs2FChoaFq1qyZpkyZ4nz/qaee0ltvvaVPPvnEeRZo5cqVkqRdu3bptttuU+3atVW3bl317t1bO3bsOGM9wPmCcAOcZwICAvSPf/xDP/74o9566y19/vnnevzxx0/bv1+/fqpfv76++eYbrV+/XiNHjlRQUJAkadu2berRo4duvvlmfffdd1qwYIHWrFmjIUOGeFRTaGioHA6HTpw4oSlTpmjixIl66aWX9N133yklJUU33HCDfv75Z0nSP/7xDy1atEjvvfeecnNz9c477ygxMbHC/a5bt06StGLFCuXn5+vDDz88pc+tt96q//znP/riiy+cbfv379fSpUvVr18/SdLq1as1YMAAPfzww9q0aZPeeOMNzZkzR88995zbx7hjxw4tW7ZMdrvd2eZwOFS/fn0tXLhQmzZt0rhx4zR69Gi99957kqQRI0botttuU48ePZSfn6/8/Hx17txZx48fV0pKisLDw7V69WqtXbtWtWrVUo8ePVRWVuZ2TYBl+fy54wCqXGpqqgkMDDQ1a9Z0brfcckuFfRcuXGj+8pe/OF+/+eabJjIy0vk6PDzczJkzp8Kx9957r7n//vtd2lavXm0CAgLM0aNHKxzzx/1v2bLFNG3a1HTo0MEYY0x8fLx57rnnXMZcdtll5qGHHjLGGDN06FDTrVs343A4Kty/JPPRRx8ZY4zJy8szksyGDRtc+qSmpprevXs7X/fu3dvcc889ztdvvPGGiY+PN+Xl5cYYY7p3724mTJjgso+3337bxMXFVViDMcakp6ebgIAAU7NmTRMSEmIkGUlm0qRJpx1jjDGDBw82N99882lrPfnZzZo1c5mD0tJSExoaapYtW3bG/QPnA9bcABbVtWtXvf76687XNWvWlPTbWYyMjAz99NNPKikp0YkTJ3Ts2DEdOXJEYWFhp+wnLS1N9913n95++23npZXGjRtL+u2S1Xfffad33nnH2d8YI4fDoby8PLVo0aLC2oqLi1WrVi05HA4dO3ZMV1xxhWbOnKmSkhLt2bNHl19+uUv/yy+/XBs3bpT02yWlq6++Ws2aNVOPHj10/fXX65prrjmruerXr58GDRqk1157TcHBwXrnnXd0++23KyAgwHmca9eudTlTU15efsZ5k6RmzZpp0aJFOnbsmP73f/9XOTk5Gjp0qEufqVOnavbs2dq5c6eOHj2qsrIytW3b9oz1bty4UVu3blV4eLhL+7Fjx7Rt27ZKzABgLYQbwKJq1qypiy66yKVtx44duv766/Xggw/queeeU926dbVmzRrde++9Kisrq/BL+qmnntKdd96pxYsX67PPPlN6errmz5+vG2+8UYcOHdLf/vY3DRs27JRxDRo0OG1t4eHhys7OVkBAgOLi4hQaGipJKikp+dPjateunfLy8vTZZ59pxYoVuu2225ScnKz333//T8eeTq9evWSM0eLFi3XZZZdp9erVevnll53vHzp0SOPHj9dNN910ytiQkJDT7tdutzv/GTz//PO67rrrNH78eD3zzDOSpPnz52vEiBGaOHGikpKSFB4erhdffFH/+te/zljvoUOH1L59e5dQedK5smgc8CfCDXAeWb9+vRwOhyZOnOg8K3FyfceZNG3aVE2bNtXw4cN1xx136M0339SNN96odu3aadOmTaeEqD8TEBBQ4ZiIiAjFx8dr7dq16tKli7N97dq16tixo0u/vn37qm/fvrrlllvUo0cP7d+/X3Xr1nXZ38n1LeXl5WesJyQkRDfddJPeeecdbd26Vc2aNVO7du2c77dr1065ubkeH+cfjRkzRt26ddODDz7oPM7OnTvroYcecvb545kXu91+Sv3t2rXTggULFB0drYiIiLOqCbAiFhQD55GLLrpIx48f1yuvvKLt27fr7bff1rRp007b/+jRoxoyZIhWrlypf//731q7dq2++eYb5+WmJ554Ql999ZWGDBminJwc/fzzz/rkk088XlD8e4899pj+/ve/a8GCBcrNzdXIkSOVk5Ojhx9+WJI0adIkvfvuu/rpp5+0ZcsWLVy4ULGxsRX+8GB0dLRCQ0O1dOlSFRYWqri4+LSf269fPy1evFizZ892LiQ+ady4cZo7d67Gjx+vH3/8UZs3b9b8+fM1ZswYj44tKSlJl1xyiSZMmCBJatKkib799lstW7ZMW7Zs0dixY/XNN9+4jElMTNR3332n3NxcFRUV6fjx4+rXr5+ioqLUu3dvrV69Wnl5eVq5cqWGDRumX375xaOaAEvy96IfAN5X0SLUkyZNmmTi4uJMaGioSUlJMXPnzjWSzK+//mqMcV3wW1paam6//XaTkJBg7Ha7iY+PN0OGDHFZLLxu3Tpz9dVXm1q1apmaNWuaSy655JQFwb/3xwXFf1ReXm6eeuopU69ePRMUFGTatGljPvvsM+f706dPN23btjU1a9Y0ERERpnv37iY7O9v5vn63oNgYY2bMmGESEhJMQECA6dKly2nnp7y83MTFxRlJZtu2bafUtXTpUtO5c2cTGhpqIiIiTMeOHc306dNPexzp6emmTZs2p7S/++67Jjg42OzcudMcO3bM3H333SYyMtLUrl3bPPjgg2bkyJEu4/bu3eucX0nmiy++MMYYk5+fbwYMGGCioqJMcHCwadSokRk0aJApLi4+bU3A+cJmjDH+jVcAAADew2UpAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKYQbAABgKf8PFNkkLZc5fegAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fpr, tpr, thresholds =roc_curve(y_test,clf.predict_proba(X_test)[:,1])\n",
    "roc_auc = metrics.auc(fpr, tpr)\n",
    "display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)\n",
    "display.plot()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "filename = 'naive_bayes.pkl'\n",
    "pickle.dump(clf, open(filename, 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
