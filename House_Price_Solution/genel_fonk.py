"""
Bu dosyada, makine öğrenmesi projelerinde sıkça ihtiyaç duyulan işlemler fonksiyonlar halinde tanımlanmıştır. 
Her projede bu fonksiyonları tekrar tekrar yazmak yerine, merkezi bir yapıdan çağırarak kod tekrarını önlemeyi ve geliştirme sürecini hızlandırmayı amaçladım.
Ayrıca, her fonksiyonun başına eklediğim açıklamalarda (docstring), fonksiyonun amacı ve parametreleri detaylı şekilde belirtilmiştir.
"""


from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
def evaluate_model_performance(model, X, y, cv=10):
    """
    Modelin performansını değerlendiren fonksiyon.
    
    Parameters:
    - model: Değerlendirilecek model (örneğin RandomForestClassifier)
    - X: Özellikler (features)
    - y: Hedef değişken (target)
    - cv: Çapraz doğrulama katman sayısı (default=10)
    
    Returns:
    - Ortalama accuracy, F1 skoru ve ROC AUC değerlerini döner.
    """
    cv_results = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    
    avg_accuracy = np.mean(cv_results["test_accuracy"])
    avg_f1 = np.mean(cv_results["test_f1"])
    avg_roc_auc = np.mean(cv_results["test_roc_auc"])
    
    print(f"Ortalama Accuracy: {avg_accuracy:.4f}")
    print(f"Ortalama F1 Skoru: {avg_f1:.4f}")
    print(f"Ortalama ROC AUC: {avg_roc_auc:.4f}")
    
    return avg_accuracy, avg_f1, avg_roc_auc


def check_df(dataframe, head=5):
    """_summary_
    Shape: Veri setinin satır ve sütun sayısını yazdırır.

    Types: Her sütunun veri tipini gösterir.

    Head: İlk head kadar satırı yazdırır.

    Tail: Son head kadar satırı yazdırır.

    NA: Hangi sütunda kaç tane eksik (NaN) değer olduğunu gösterir.

    Quantiles: Sayısal sütunlar için seçilen yüzdelik dilimleri (0%, 5%, 50%, 95%, 99%, 100%) hesaplayıp gösterir.

    Args:
        dataframe (_type_): _description_
        head (int, optional): _description_. Defaults to 5.
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include='number').quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    

def grap_col_names(dataframe, cat_th=10, car_th=20):
    """
    DataFrame'deki değişkenleri kategorik, numerik ve kardinal (yüksek kardinaliteli kategorik) olarak ayırır.

    Args:
        dataframe (pd.DataFrame): Değişkenleri ayrıştırılacak veri çerçevesi.
        cat_th (int, optional): Numerik olup kategorik gibi değerlendirilecek değişkenler için eşik değeri. Varsayılan: 10.
        car_th (int, optional): Kategorik olup kardinal (çok benzersiz sınıfa sahip) sayılacak değişkenler için eşik. Varsayılan: 20.

    Returns:
        tuple: 
            - cat_cols (list): Kategorik değişkenlerin listesi.
            - num_cols (list): Numerik değişkenlerin listesi.
            - cat_but_car (list): Kardinal kategorik değişkenlerin listesi.
    """

    
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Sayısal değişkenler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    """
    Verilen kategorik değişkenin sınıf dağılımını ve oranlarını gösterir. 
    İsteğe bağlı olarak sütunun grafiğini de çizer.

    Args:
        dataframe (pd.DataFrame): İncelenecek veri çerçevesi.
        col_name (str): Analiz edilecek kategorik sütunun adı.
        plot (bool, optional): Sütun için countplot çizilsin mi? Varsayılan: False.

    Returns:
        None
    """


    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))
    print("##########################################")

    # Eğer plot=True ise, sütun için sayım grafiği (countplot) çizilir
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
        
        
def num_summary(dataframe, numerical_col, plot=False):
    """
    Sayısal bir değişkenin betimsel istatistiklerini ve isteğe bağlı olarak histogram grafiğini gösterir.

    Args:
        dataframe (pd.DataFrame): İncelenecek veri çerçevesi.
        numerical_col (str): Analiz edilecek sayısal sütunun adı.
        plot (bool, optional): Histogram çizilsin mi? Varsayılan: False.

    Returns:
        None
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

def target_summary_with_num(dataframe, target, numerical_col):
    """
    Hedef değişkenin (target) her bir sınıfı için sayısal bir değişkenin ortalamasını hesaplar ve gösterir.

    Args:
        dataframe (pd.DataFrame): Analiz yapılacak veri çerçevesi.
        target (str): Bağımlı (hedef) değişkenin sütun adı.
        numerical_col (str): Ortalama hesaplanacak sayısal değişkenin sütun adı.

    Returns:
        None
    """
    
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    """
    Kategorik bir değişkenin her bir sınıfı için hedef değişkenin ortalamasını, sınıf sayısını ve oranını hesaplar ve yazdırır.

    Args:
        dataframe (pd.DataFrame): Analiz yapılacak veri çerçevesi.
        target (str): Bağımlı (hedef) değişkenin sütun adı (örneğin: 0/1).
        categorical_col (str): İncelenecek kategorik bağımsız değişkenin adı.

    Returns:
        None
    """

    
    print(categorical_col)
    print(pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
        "Count": dataframe[categorical_col].value_counts(),
        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    }), end="\n\n\n")


def missing_values_table(dataframe, na_name=False):
    """
    Veri setindeki eksik değerleri içeren sütunları, bu sütunlardaki eksik değer sayısını ve oranlarını tablo olarak gösterir.
    
    Args:
        dataframe (pd.DataFrame): Eksik değer kontrolü yapılacak veri çerçevesi.
        na_name (bool, optional): True olarak ayarlanırsa eksik değer içeren sütunların isimlerini liste olarak döner. Varsayılan: False.

    Returns:
        list (optional): Eksik değer içeren sütunların isim listesi (eğer na_name=True ise).
    """

    
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Belirtilen kategorik değişkenleri one-hot encoding yöntemi ile ikili (binary) sütunlara çevirir.

    Args:
        dataframe (pd.DataFrame): İşlem yapılacak veri çerçevesi.
        categorical_cols (list): One-hot encoding uygulanacak kategorik değişkenlerin isim listesi.
        drop_first (bool, optional): İlk kategoriyi düşürerek multikolinearliteyi azaltmak için kullanılır. Defaults to False.

    Returns:
        pd.DataFrame: One-hot encoding uygulanmış yeni veri çerçevesi.
    """

    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    return dataframe

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Belirtilen sütun için alt ve üst aykırı değer eşiklerini hesaplar.
    Varsayılan olarak %5 ve %95 çeyreklikleri kullanır (esnek yapı).

    Args:
        dataframe (pd.DataFrame): Aykırı değer analizi yapılacak veri çerçevesi.
        col_name (str): Eşik hesaplaması yapılacak sayısal değişkenin sütun adı.
        q1 (float, optional): Alt çeyreklik (default: 0.05).
        q3 (float, optional): Üst çeyreklik (default: 0.95).

    Returns:
        tuple: (alt sınır, üst sınır) değerleri. Bu sınırların dışında kalanlar aykırı kabul edilir.
    """

    
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)

    
    interquantile_range = quartile3 - quartile1

    
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Belirtilen sütunda aykırı değer (outlier) olup olmadığını kontrol eder.

    Args:
        dataframe (pd.DataFrame): Kontrol edilecek veri çerçevesi.
        col_name (str): Aykırı değer kontrolü yapılacak sütun adı.

    Returns:
        bool: Eğer sütunda aykırı değer varsa True, yoksa False döner.
    """

    
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
    Belirtilen sütundaki aykırı değerleri alt ve üst sınırlar ile baskılar (cap/floor işlemi).

    Args:
        dataframe (pd.DataFrame): Aykırı değer baskılaması yapılacak veri çerçevesi.
        variable (str): Aykırı değer kontrolü yapılacak sütun adı.
        q1 (float, optional): Alt çeyreklik değeri (default: 0.05).
        q3 (float, optional): Üst çeyreklik değeri (default: 0.95).

    Bu yöntem, uç (outlier) değerleri, belirlenen eşikler ile sınırlar ve veri setinden çıkarmaz.
    """

    
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def label_encoder(dataframe, binary_col):
    """
    İkili (binary) kategorik bir sütunu 0 ve 1 şeklinde sayısal değerlere dönüştürür.

    Args:
        dataframe (pd.DataFrame): Etiketleme yapılacak veri çerçevesi.
        binary_col (str): İkili değerlere sahip sütun adı (örneğin: "Evet"/"Hayır", "Erkek"/"Kadın" gibi).

    Returns:
        pd.DataFrame: Etiketlenmiş (label encoded) veri çerçevesi.
    """

    
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    Verilen kategorik değişkenleri one-hot encoding yöntemiyle 0 ve 1'den oluşan dummy değişkenlere dönüştürür.

    Args:
        dataframe (pd.DataFrame): Dönüştürme yapılacak veri çerçevesi.
        categorical_cols (list): One-hot encoding yapılacak kategorik sütunların listesi.
        drop_first (bool, optional): Multikolinearliteyi azaltmak için ilk kategoriyi düşür (default: False).

    Returns:
        pd.DataFrame: Dummy değişkenlerle genişletilmiş yeni veri çerçevesi.
    """

    
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

    return dataframe



def plot_correlation_matrix(dataframe, numeric_columns, title="Correlation Matrix", cmap="magma", figsize=(18, 13)):
    """
    Verilen sayısal sütunlar arasındaki korelasyonları ısı haritası ile görselleştirir.

    Parametreler:
        dataframe (pd.DataFrame): Veri çerçevesi
        numeric_columns (list): Sayısal değişkenlerin isimleri
        title (str): Grafik başlığı
        cmap (str): Renk skalası (örn: 'magma', 'coolwarm', 'viridis')
        figsize (tuple): Grafik boyutu

    Returns:
        None (plt.show() ile gösterim yapılır)
    """
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dataframe[numeric_columns].corr(), annot=True, fmt=".2f", ax=ax, cmap=cmap)
    ax.set_title(title, fontsize=20)
    plt.show()
