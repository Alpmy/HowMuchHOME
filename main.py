import time
import pandas as pd
import numpy as np
from playwright.sync_api import sync_playwright
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

def get_data():
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else context.new_page()

        all_data = []
        page_num = 1

        while True:
            print(f"--- Sayfa {page_num} taranıyor ---")
            page.wait_for_selector(".searchResultsItem")

            rows = page.locator("tr.searchResultsItem:not(.nativeAd):not(.searchResultsPromoTop)").all()

            for row in rows:
                try:
                    title_elem = row.locator("td.searchResultsTitleValue a.classifiedTitle")
                    title = title_elem.inner_text().strip() if title_elem.count() > 0 else ""

                    # Satırdaki tüm özellik hücreleri (m2, oda sayısı vb.)
                    attr_elems = row.locator("td.searchResultsAttributeValue")

                    # İlk hücre m² (en az 1 hücre varsa)
                    m2 = attr_elems.nth(0).inner_text().strip() if attr_elems.count() > 0 else ""

                    # İkinci hücre Oda Sayısı (en az 2 hücre varsa)
                    oda_sayisi = attr_elems.nth(1).inner_text().strip() if attr_elems.count() > 1 else ""

                    price_elem = row.locator("td.searchResultsPriceValue span")
                    price = price_elem.inner_text().strip() if price_elem.count() > 0 else ""

                    if title:
                        all_data.append({
                            "Baslik": title,
                            "m2": m2,
                            "Oda_Sayisi": oda_sayisi,
                            "Fiyat": price
                        })
                except Exception:
                    continue

            print(f"Şu ana kadar toplanan toplam ilan: {len(all_data)}")

            # "Sonraki" sayfa butonu var mı kontrol et
            next_button = page.locator("a.prevNextBut[title='Sonraki']")
            if next_button.count() > 0 and next_button.is_visible():
                next_button.click()
                page_num += 1
                time.sleep(3)  # Bot engeline takılmamak için insani bekleme süresi
            else:
                print("Son sayfaya ulaşıldı veya başka sayfa yok.")
                break

        df = pd.DataFrame(all_data)
        df.to_csv("mahalle_tum_ilanlar.csv", index=False, encoding="utf-8-sig")
        print(f"İşlem tamamlandı! Toplam {len(df)} adet veri kaydedildi.")

def data_split_class(data):
    median = data["Fiyat"].median()
    median=input("Eşik Değer Kaç:")
    median=int(median)
    print(f"Otomatik seçilen medyan eşik: {median} TL \n")

    data = data.assign(
        target=np.where(data["Fiyat"] < median, 0, 1)
    ).drop(["Fiyat", "Baslik", "Oda_Sayisi"], axis=1, errors="ignore")

    data = data.dropna(subset=["m2", "oda", "salon", "target"])

    X = data[["m2", "oda", "salon"]]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test, median

def std_fit(X_train, X_test, y_train, y_test):
    std = StandardScaler()

    X_train_std = std.fit_transform(X_train[["m2", "oda","salon"]])
    X_test_std = std.transform(X_test[["m2", "oda","salon"]])

    max_k = min(len(X_train) - 1, 30)
    parameters = {"n_neighbors": range(1, max_k, 2)}

    gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)
    gridsearch.fit(X_train_std, y_train)

    k=gridsearch.best_params_["n_neighbors"]

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)

    print(knn.score(X_test_std, y_test))
    print(f"Train Accuracy: {accuracy_score(y_train, knn.predict(X_train_std))}")
    print(f"Test Accuracy: {accuracy_score(y_test, knn.predict(X_test_std))}")

    return knn,std

def analyze():
    df = pd.read_csv("mahalle_tum_ilanlar.csv")
    pd.set_option('display.max_columns', None)

    pd.set_option('display.width', 1000)

    #print(df.head())
    #print(df.dtypes)

    dfNew=df
    dfNew["Fiyat"]=(df["Fiyat"].str.replace(" TL","",regex=False)
                        .str.replace(".","", regex=False)
                        .str.strip()
                        .astype("Int64"))

    dfNew[['oda', 'salon']] = df['Oda_Sayisi'].str.split('+', expand=True)

    dfNew["oda"]=pd.to_numeric(dfNew["oda"])
    dfNew["salon"]=pd.to_numeric(dfNew["salon"])

    #print(dfNew.dtypes)

    dfNew["oda"] = pd.to_numeric(dfNew["oda"], errors="coerce").fillna(1)
    dfNew["salon"] = pd.to_numeric(dfNew["salon"], errors="coerce").fillna(0)

    return dfNew

def prob(knn,std):
    print()

    oda=input("Oda Sayısı:")
    salon=input("Salon Sayısı:")
    m2=input("Ev Kaç m2:")

    your_home=pd.DataFrame([[m2, oda, salon]], columns=["m2", "oda", "salon"])
    your_home_std=std.transform(your_home[["m2", "oda", "salon"]])

    result_proba =knn.predict_proba(your_home_std)[0]
    result=knn.predict(your_home_std)[0]

    return result_proba, result

def printf(proba, result,median):
    print("\nEviniz İçin Yapılan Tahminin Sonucu:")

    if result == 1:
        print(f"{median}TL'den yüksek olmalı | Olasılık:{proba[1]}")
    if result == 0:
        print(f"{median}TL'den düşük olmalı | Olasılık:{proba[0]}")

def main():
    #get_data() #dataları almak için
    df=analyze()

    print("\nAnaliz bitti sonuçlar...")

    X_train, X_test, y_train, y_test, median = data_split_class(df)
    knn,std=std_fit(X_train, X_test, y_train, y_test)

    proba,guess=prob(knn,std)

    printf(proba,guess,median)

if __name__ == "__main__":
    main()

