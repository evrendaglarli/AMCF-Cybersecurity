# AMCF — UNSW-NB15 gerçek veri üzerinde çoklu-tohum sağlamlık değerlendirmesi

**19 Ekim 2026 — 5 tohum, gerçek veri, AMCF-lite. Bu rapor IEEE Access R20'nin deneylerini yeniden üretmez.**

## Kullanılan veriler ve kapsam

Kullanıcının sağladığı `UNSW_NB15_training_and_testing-set.zip` arşivindeki 175.341 eğitim ve 82.332 test kaydı. Her iki dosya 45 sütun içeriyor; kayıt sayıları akademik bölümlerle uyumlu. İndirme kaynağıyla bağımsız imzalı/hash kıyaslaması gerçekleştirilmedi.

- Eğitim CSV SHA-256: `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa`
- Test CSV SHA-256: `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559`
- 5 tohum: **42, 43, 44, 45, 46**; her tohumda resmî eğitim bölümünden **20.000 stratified kayıt** seçildi; **82.332 test kaydının tamamı** kullanıldı.
- Seçilen 20.000 kaydın yaklaşık %70'i model eğitimi, %15'i isotonic calibration, %15'i FPR operating-point belirleme için ayrıldı. Test bölümü yalnızca sonuç hesaplamada kullanıldı.
- Hedef doğrulama FPR değerleri: **%1, %5, %10**. Seçilen eşikler test etiketleri kullanılarak yeniden ayarlanmadı.
- Model 1: Logistic Regression; model 2: HistGradientBoosting; model 3: **AMCF-lite**, bu iki modelin kalibre olasılıklarının eşit ağırlıklı logit füzyonu ve sanal ignore/escalate maliyet eşiği.
- AMCF-lite: **gerçek AMCF değildir**; bellek, hipotez/coherence, VoI, drift uyarlama, SOAR, gerçek güvenlik kapısı veya containment yoktur.

## Beş tohumun ortalama sonuçları

Değerler aynı resmî test bölümü üzerindeki beş farklı eğitim bölünmesinin ortalamasıdır; bağımsız dış doğrulama değildir.

| Doğrulama-FPR hedefi | Model | Ortalama Recall | Ortalama test FPR | Ortalama ROC-AUC | Recall aralığı (tohumlar) |
|---|---|---:|---:|---:|---:|
| %1 | Logistic Regression | %72,15 | %0,87 | 0,9521 | %71,20–73,40 |
| %1 | Gradient Boosting | %83,15 | %1,29 | 0,9808 | %77,27–85,98 |
| %1 | AMCF-lite | %82,13 | %1,44 | 0,9759 | %80,21–83,79 |
| %5 | Logistic Regression | %83,72 | %8,48 | 0,9521 | %77,61–87,62 |
| %5 | Gradient Boosting | **%94,34** | **%11,52** | **0,9808** | %93,23–95,59 |
| %5 | AMCF-lite | %93,07 | %13,06 | 0,9759 | %91,35–94,82 |
| %10 | Logistic Regression | %90,22 | %20,61 | 0,9521 | %88,45–91,75 |
| %10 | Gradient Boosting | **%97,41** | **%20,47** | **0,9808** | %97,15–98,03 |
| %10 | AMCF-lite | %96,96 | %23,70 | 0,9759 | %95,90–97,76 |

**%5 hedefinde AMCF-lite − Gradient Boosting:** ortalama recall farkı **−1,27 yüzde puanı**, FPR farkı **+1,54 yüzde puanı**. Beş tohumun ikisinde AMCF-lite FPR daha düşüktür ve birinde recall daha yüksektir; bu nedenle tek tek koşular tek yönlü değildir. Ortalama bulgu AMCF-lite üstünlüğünü desteklemez.

**Önemli:** Doğrulama verisindeki %5 hedefi ile bağımsız test FPR'si aynı şey değildir. %5 hedefinde test FPR ortalamaları %8,48 / %11,52 / %13,06 olarak gerçekleşmiştir. Bu, mevcut eşik belirleme yaklaşımının test dağılımında nominal hata bütçesini kontrol edemediğini gösterir. Test verisinden eşik seçerek düzeltme yapılmadı.

## Güven aralıkları — kapsamı

- Her **tekil tohum** için recall ve test FPR'sinin **Wilson %95 oran güven aralıkları** `seed_operating_points.csv` dosyasındadır.
- **Eşleştirilmiş ve sınıf-koşullu 2.000 yeniden örneklemeli bootstrap** ile AMCF-lite ile her baseline arasındaki recall/FPR farklarının %95 aralıkları `paired_differences.csv` dosyasındadır.
- Bu güven aralıkları sabit test bölümündeki kayıtların koşullu yeniden örneklenmesine dayanır. **Olaylar arasındaki zamansal/host/incident korelasyonunu modellemezler**; dar aralıklar gerçek SOC ortamına genellenebilirlik veya model üstünlüğü anlamına gelmez.
- **Beş tohum birbirinden bağımsız test veri kümeleri değildir**; tohumlar arası değişim model öğreniminin örneklem duyarlılığıdır, dış doğrulama veya meta-analiz değil. Kaydedilen aralıklar tek bir tohumun test veri örneklemesi içindir; tohum ortalamalarına ait birleşik %95 CI değildir.
- p-değeri, düzeltilmiş çoklu test anlamlılığı, BCa, DeLong veya blok bootstrap iddia edilmiyor.

## Dosyalar ve tekrarlama

- `experiments/unsw_nb15_robustness.py`: beş tohum, üç FPR hedefi, güven aralıkları ve kayıt üretimi
- `tests/test_unsw_nb15_robustness.py`: 5 matematik/sözleşme birim testi
- `results/robustness/seed_operating_points.csv`: 45 satır; her tohum / bütçe / model
- `results/robustness/paired_differences.csv`: 30 satır; karşılaştırma başına fark ve bootstrap aralığı
- `results/robustness/seed_summary.csv`: 9 satırlık ortalama ve tohum aralığı özeti
- `results/robustness/provenance.json`: girdi hash, sürümler, bölüm sayıları, yöntem ve sınırlar

```bash
python -m pip install numpy pandas scipy scikit-learn
python -m unittest discover -s tests -p 'test_unsw_nb15_robustness.py' -v
python experiments/unsw_nb15_robustness.py \
  --train data/raw/unsw_nb15/UNSW_NB15_training-set.csv \
  --test data/raw/unsw_nb15/UNSW_NB15_testing-set.csv \
  --out results/robustness --max-train 20000 \
  --seeds 42 43 44 45 46 --budgets 0.01 0.05 0.10 \
  --bootstrap 2000
```

Not: Basit pilotta `unsw_nb15_feasibility.py` dosyası aynı `experiments/` klasöründe olmalı. Gerçek ham CSV dosyaları GitHub'a eklenmedi.

## Bilimsel karar

**Yalnızca AMCF-lite fizibilite deneyi için** uçtan uca veri okuma, öğrenme, kalibrasyon, eşik seçimi ve doğrulama altyapısı çalışıyor. Daha iyi recall gösterdiği bazı tohumlarda FPR cezası bulunmaktadır; Gradient Boosting'e karşı tekrarlanabilir üstünlük saptanmadı. Tam AMCF, olay temelli saldırı zincirleri, 60 günlük SOC deneyleri, containment ve 12 katılımcılı human-factors değerlendirmesi bunların kapsamı dışındadır. **R20'nin bildirilmiş sayısal sonuçları bu pilotla teyit edilmez; R20 üzerinde otomatik bir düzeltme yapılmadı.**