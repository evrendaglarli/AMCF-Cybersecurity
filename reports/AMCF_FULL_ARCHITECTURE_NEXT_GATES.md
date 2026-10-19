# AMCF tam mimariye geçiş — kabul kapıları (R20 doğrulaması değildir)

**Bağlayıcı ilke:** UNSW'nin resmî eğitim/test tablolarında uzun süreli attack-chain aşaması, SOC'da gerçekleşmiş containment ve analist değerlendirmesi gibi kanıtlar yok. Kod geliştirmek bu verileri üretmez. Yeni deneyler, yeni deney olarak ve elde edilen gerçek sayılarla raporlanır.

## Gate 0 — çalışan ve güvenli referans çekirdeği

- Düzeltilen modüllerin import, tip, boyut ve konfigürasyon sözleşmeleri test edilmeli (`working_memory.Event` circular import, 384/768 embedding uyumsuzluğu, `SafetyGate.check()` veri tipi, TCN giriş boyutları, `setup.py` şablon artıkları).
- Özerk SOAR/actuation **kapalı**; güvenlik kapısı kanıtı olmadan otomasyon yapılmamalı. SafetyGate içinde az örnek olduğunda 'assume safe' davranışı kaldırılıp fail-closed yapılmalı.
- Girdi kaynakları ve eğitim/kalibrasyon/test izolasyonu sistem düzeyinde korunmalı; çevrimiçi label feedback yalnızca etiketin öğrenilebilir hâle geldiği event-time'da kullanılmalı.
- Kabul: kod çalışıyor, bütün testler geçiyor; hareket/otomasyon kapalı, test-temelli leakage kontrolleri mevcut.

## Gate 1 — gerçek hipotez ve coherence durumu

- MITRE ATT&CK TTP eşlemesi ve saldırı hipotezi şablonları **versiyonlu, harici veri** olmalı; `UNSW` etiketi `attack_cat` testten modele sızdırılmamalı.
- Entity-based bounded hypothesis set, PathCons, StageCov, Rate bileşenleri ve Γ ağırlıkları, Eq. (4) ve (5) ile uyumlu bellek durumu ayrı test edilmeli.
- Evidence de-duplication, top-K pruning, stale expiry, per-hypothesis once-only update ve normalization invariants ile unit/integration testleri yazılmalı.
- Kabul: yapay fixture ile yapılan unit testleri ve tarihli gerçek ATT&CK-aligned event stream üzerinde sistem testi; bu veri olmadan 'long-dwell recall' iddiası yok.

## Gate 2 — operasyonel meta-kontrol

- VoI için gerçek enrichment sonuçlarının koşullu dağılımı, olay gecikmesi ve maliyet birimleri doğrulanmalı; random posterior kaydırma kaldırılmalı.
- Online PSI/KS/ECE drift detection ve validasyonu; mevcut modelin risk eşikleri ve calibration drift kararları entegre edilmeli.
- Baseline FPR/hour hedefi için gerçek zaman damgalı log replays gerekir; statik UNSW flow verilerinde yalnızca FPR raporlanabilir.
- Kabul: kontrollü veri üzerinde önceden belirlenmiş metrikler ve drift ablations; ROC/AUC tek başına yeterli değil.

## Gate 3 — deney protokolü ve istatistikler

- Aynı olaylar üzerinde adil SIEM, ML, AMCF ve frozen kontrol varyantları.
- Incident/host/time block bootstrap, calibrated fused posterior reliability, önceden yazılmış karşılaştırmalar, Holm düzeltmesi, DeLong testi (gerçek algoritma).
- Güvenlik: koşullu bağımsız kanıt için gözlenen logit korelasyon; zero-event bootstrap risk bound değil; az sayıda güvenlik olayı olduğunda fail-closed.
- Kabul: her tablo/şekil için commit, seed, veri hash, script, environment, tekil run manifesti ve bağımsız doğrulama raporu.

## Gate 4 — makaleye aktarım için ayrı karar

- Gerçek veri farklı sayılar verirse R20 metnindeki sayılar aynen tutulup yeniden üretilmiş gibi sunulamaz.
- Enterprise SOC kayıtları ve insan katılımcı verileri yoksa ilgili ölçümler özgün kanıt olmadan 'doğrulandı' denilemez. Bunlar için yeni, izinli deneyler gerekir.
- Sadece tamamlanmış doğrulanabilir bir gate sonrası bir sonraki gate'e geçilsin. Bütün aşamalar aynı PR'a yığılmasın.