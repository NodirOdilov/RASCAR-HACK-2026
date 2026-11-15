# Файлы отправки — команда «Эволюция», RASCAR.HACK 2026, кейс 1

Все четыре основных файла имеют одинаковый формат: 10 869 байт, заголовок `id,score`, 362 строки в порядке
`sample_submission.csv`, без дублей и пропусков, оценки конечны и лежат в [0, 1], переводы строк LF, без BOM.
У каждого файла есть файл `.sha256` и манифест JSON с происхождением, локальной проверкой и командой воспроизведения.

| файл | модель | SHA-256 | «день на день», среднее | Kaggle Public AP |
| --- | --- | --- | ---: | ---: |
| **`EVOLYUTSIYA_CASE1_E4_OPTIONAL.csv`** | **E4** `e4_e2x_one_to_one` — итоговая модель | `7f4c2569b820b899a4a9dd7621eb662369d2c50bef5e0a9137954667ec9c210a` | 0.9742 | **0.95778** |
| `EVOLYUTSIYA_CASE1_E2_CANDIDATE.csv` | E2 `e2_smooth_hires_small` | `75db6eef0d57b24aca59f8dd89e8917be971850426e83862dd04ca051545021b` | 0.9666 | 0.93571 |
| `EVOLYUTSIYA_CASE1_E2X_OPTIONAL.csv` | E2X `e2x_pixel_inner_hires` (база E4) | `8ab7cb3ec8bff39018ff1063b6544a8b8293c084259c46f06642abf0cc57bdf0` | 0.9702 | 0.93284 |
| `EVOLYUTSIYA_CASE1_E1_CANDIDATE.csv` | E1 `e1_smooth_pooled` | `0aab6ce8a0f5a7170cf8b1fc43fcf19e892423cf828674130ba6500c3541cc63` | 0.9654 | 0.92964 |

Имена файлов сохранены такими, под какими файлы создавались и загружались (слова `OPTIONAL` и `CANDIDATE` отражают
статус на момент создания), а содержимое не менялось: SHA-256 совпадают с файлами `.sha256` и манифестами.

**Манифесты — снимок на момент создания файлов (13–14 ноября 2026), до загрузки E4 и E2 на Kaggle.** Поэтому в
`E4_OPTIONAL_MANIFEST.json` и `E2_CANDIDATE_MANIFEST.json` стоит `"uploaded": false`, а в описаниях E4 названа
необязательным вторым файлом. Актуальные результаты Public приведены в таблице выше и в [`../README.md`](../README.md),
§11. Результаты Private неизвестны. Пути `results/…` и `research/…` в манифестах относятся к рабочей копии команды.

`kaggle_submitted/` содержит байтовые копии двух ранних загруженных файлов от 12 ноября: `vf_blend_geo_gbm`, Public
0.71980, и `c9_lr_geo`, Public 0.80490. Они нужны скрипту `research/lead/v3_export_frozen.py`, который добавляет их
оценки в таблицу признаков как справочные столбцы.

Проверить контрольную сумму:

```powershell
Get-FileHash submissions\EVOLYUTSIYA_CASE1_E4_OPTIONAL.csv -Algorithm SHA256
```

```bash
sha256sum submissions/EVOLYUTSIYA_CASE1_*.csv    # сравнить с таблицей выше
```

Файл `.gitattributes` отключает преобразование концов строк в git, поэтому после клонирования суммы совпадают на
любой платформе.
