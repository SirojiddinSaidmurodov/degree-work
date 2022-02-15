[Введение [3](#введение)](#введение)

[1. Формирование требований
[7](#формирование-требований)](#формирование-требований)

[1.1. Изучение предметной области машинного обучения и нейронных сетей
[7](#изучение-предметной-области-машинного-обучения-и-нейронных-сетей)](#изучение-предметной-области-машинного-обучения-и-нейронных-сетей)

[1.2. Анализ йподходов к обработке естественного языка
[7](#анализ-подходов-к-обработке-естественного-языка)](#анализ-подходов-к-обработке-естественного-языка)

[1.3. Техническое задание
[7](#техническое-задание)](#техническое-задание)

[2. Проектирование платформы для обучения нейросетевой модели
[8](#проектирование-платформы-для-обучения-нейросетевой-модели)](#проектирование-платформы-для-обучения-нейросетевой-модели)

[2.1. Используемые модели нейронных сетей
[8](#используемые-модели-нейронных-сетей)](#используемые-модели-нейронных-сетей)

[2.2. Формализация алгоритмов основного анализа текста
[8](#формализация-алгоритмов-основного-анализа-текста)](#формализация-алгоритмов-основного-анализа-текста)

[2.3. Структура платформы для обучения нейросетевой модели
[8](#структура-платформы-для-обучения-нейросетевой-модели)](#структура-платформы-для-обучения-нейросетевой-модели)

[3. Экспериментальная реализация алгоритма анализа текста
[9](#экспериментальная-реализация-алгоритма-анализа-текста)](#экспериментальная-реализация-алгоритма-анализа-текста)

[3.1. Подготовка данных для обучения
[9](#подготовка-данных-для-обучения)](#подготовка-данных-для-обучения)

[3.2. Реализация платформы для обучения нейросетевой модели
[9](#реализация-платформы-для-обучения-нейросетевой-модели)](#реализация-платформы-для-обучения-нейросетевой-модели)

[3.3. Результаты машинного обучения
[9](#результаты-машинного-обучения)](#результаты-машинного-обучения)

[4. Тестирование по результатам обучения модели для основного анализа
текста
[10](#тестирование-по-результатам-обучения-модели-для-основного-анализа-текста)](#тестирование-по-результатам-обучения-модели-для-основного-анализа-текста)

[4.1. Экспериментальная оценка результатов обучения
[10](#экспериментальная-оценка-результатов-обучения)](#экспериментальная-оценка-результатов-обучения)

[4.2. Целевой оценочный тест реализации алгоритма
[10](#целевой-оценочный-тест-реализации-алгоритма)](#целевой-оценочный-тест-реализации-алгоритма)

[Заключение [11](#заключение)](#заключение)

[Список использованных источников [12](#_Toc94970231)](#_Toc94970231)

# Введение {#введение .Title}

Для понимания текста людьми и программами искусственного интеллекта (ИИ)
часто требуется правильное использование заглавных букв и знаков
препинания. Простые предложения, состоящие из нескольких слов, и люди, и
ИИ могут читать и обрабатывать, даже если использование заглавных букв
или знаков препинания отсутствует или неправильно. Например, в случае
голосовых команд обработка обычно выполняется только для распознанных
слов нижнего регистра. Однако, если объем текста, подлежащего анализу,
увеличивается, например, до целых абзацев или страниц, то даже для
человека быстрое понимание его смысла становится сложной задачей. Это
было изучено Джонсом и др. (2003), которые проанализировали влияние
заглавных букв и пунктуации на читаемость расшифровки речи в текст.

Ранние работы рассматривали пунктуацию только как подсказки с точки
зрения читателя к возможным просодическим характеристикам и паузам
текста (Markwardt, 1942). Нанберг (1990) утверждает, что пунктуация
играет гораздо большую роль. Кроме того, знаки препинания
классифицируются как разграничивающие, разделяющие и устраняющие
неоднозначность. Некоторые метки, такие как запятая, могут принадлежать
к нескольким категориям, поскольку они могут выполнять несколько ролей.
Джонс (1994) доказывает, что «для более длинных предложений реального
языка грамматика, использующая знаки препинания, значительно превосходит
аналогичную грамматику, которая их игнорирует». Основываясь на этом и
других подобных выводах, современные языковые модели считают пунктуацию
частью своего словарного запаса. Сюда входят новейшие модели, такие как
BERT, ELMo, OpenAI GPT-2 и GPT-3.

Алгоритмы обработки естественного языка (NLP -- Natural language
processing), такие как распознавание именованных сущностей (NER -- Named
entitiy recognition), идентификация части речи, анализ зависимостей,
машинный перевод (MT -- Machine translation), используют заглавные буквы
в качестве признаков обрабатываемого в данный момент слова, в то время
как пунктуация используется. как признаки для соседних слов. Например,
Stanford Named Entity Recognizer рассматривает признаки на основе формы
слова. Это означает построение представления слова на основе типа
символов, встречающихся в слове. Было предложено несколько алгоритмов
представления формы слова, но общая идея состоит в том, чтобы
закодировать прописную букву определенным символом, скажем, «X»,
строчную букву «x» и цифру «d». В этом случае слово типа «McDonald»
станет «ХхХхххх». Работа любых таких алгоритмов возможна только в том
случае, если слова правильно представлены в виде прописных и строчных
букв.

Особое внимание следует уделить системам автоматического распознавания
речи (ASR -- Automatiс speech recognition). Первичный вывод таких систем
обычно состоит из необработанного текста с использованием одного и того
же регистра (нижнего, либо верхнего регистра) и без знаков препинания. В
таких ситуациях перед применением дальнейших алгоритмов NLP требуется
дополнительная предварительная обработка, чтобы восстановить правильный
регистр букв и пунктуацию. Их иногда называют «богатыми транскрипциями».
Одна из первых инициатив, касающихся автоматической расширенной
транскрипции разговорной речи, началась в 2002 году в контексте
программы DARPA «Эффективное, доступное повторное использование речи в
текст» (EARS -- Effective, Affordable, Reusable Speech-to-text), целью
которой было улучшение уровня развития алгоритмов обработки языка. С
этой целью NIST (National institute of Standards and Technology)
выпустил серию обширных наборов данных для оценки транскрипции, чтобы
помочь в оценке таких систем.

Несмотря на то, что большой объем данных, требующих восстановления
заглавных букв и пунктуации, поступает из систем ASR, необходимо также
учитывать и другие источники. Миллер и др. (2000) идентифицируют другие
источники шума в виде текста, полученного с помощью оптического
распознавания символов (OCR), или в некоторых газетных статьях. В этих
случаях отсутствие надлежащей буквы или пунктуации затрагивает не весь
текст, а его части. В случае OCR некоторые знаки препинания могут быть
не распознаны, в то время как в случае некоторых статей первое
предложение или абзац могут быть написаны только заглавными буквами.
Кроме того, в случае коротких текстовых сообщений (SMS), чатов, твитов
или других действий в микроблогах люди также могут игнорировать
правильный регистр и пунктуацию

Одна из трудностей, при создании человеко-компьютерных интерфейсов с
использованием естественного языка, с которыми приходится сталкиваться,
связана с непоследовательным использованием пользователем пунктуации и
использования заглавных букв. В этом контексте многие подходы пытаются
скрыть проблему, удаляя все знаки препинания и заглавные буквы как из
данных обучения, так и из входных данных полученных во время работы.
Кроме того, Coniam (2014) также проанализировал вывод чат-ботов с точки
зрения человека, использующего эти программы для изучения английского
как второго языка. Он смог определить проблемы с заглавными буквами и
пунктуацией даже в произведенном тексте. Тем не менее, он утверждает,
что для коротких предложений, создаваемых чат-ботами, «переход на
английский язык за счет все более широкого распространения текстовых
сообщений делает спорным вопрос о том, можно ли считать эти проблемы
важными в наши дни».

# Формирование требований

## Изучение предметной области машинного обучения и нейронных сетей

Итак, разберемся что же такое машинное обучение. В Википедии можно найти
следующее определение: «Машинное обучение --- класс методов
искусственного интеллекта, характерной чертой которых является не прямое
решение задачи, а обучение за счёт применения решений множества сходных
задач». Так же можно найти более современное определение машинного
обучения, данное Томом Митчеллом: «A computer program is said to learn
from experience E with respect to some class of tasks T and performance
measure P if its performance at tasks in T, as measured by P, improves
with experience E.». Различают два основных типа обучения:

-   Дедуктивное обучение, или обучение с учителем, предполагает
    формализацию знаний экспертов и их перенос в компьютер в виде базы
    знаний

-   Обучение по прецедентам, или индуктивное обучение (обучение без
    учителя), основано на выявлении эмпирических закономерностей в
    данных.

В первом случае важна правильно составленная и размеченная обучающая
выборка данных. Для этого необходимо выделить важные признаки данных, и
заранее определить правильный ожидаемый результат для каждого набора
данных. Дедуктивное обучение принято относить к области экспертных
систем, поэтому иногда под машинным обучением понимают дедуктивное
обучение.

Второму случаю присуще обучение на основе неразмеченных данных, и целью
алгоритмов является определение признаков данных и их приоритетов.
Многие методы, применяемые в обучении без учителя тесно связаны с
извлечением информации и анализом данных.

## Анализ подходов к обработке естественного языка

Asda

## Техническое задание

Asdada

# Проектирование платформы для обучения нейросетевой модели

Asdada

## Используемые модели нейронных сетей

Asdada

## Формализация алгоритмов основного анализа текста

фвфвф

## Структура платформы для обучения нейросетевой модели

Текст рыба

# Экспериментальная реализация алгоритма анализа текста

Текст рыба

## Подготовка данных для обучения

Текст рыба

## Реализация платформы для обучения нейросетевой модели

Текст рыба

## Результаты машинного обучения

рыыыба

# Тестирование по результатам обучения модели для основного анализа текста

Рыба

## Экспериментальная оценка результатов обучения

Рыба

## Целевой оценочный тест реализации алгоритма

# Заключение {#заключение .Title}

Рыба

# Список использованных источников {#список-использованных-источников .list-paragraph}

**Mitchell Tom M.** Machine Learning \[Книга\]. - \[б.м.\] : McGraw
Hill, 1997. - стр. 2.

**Păiş Vasile и Tufis Dan** Capitalization and Punctuation Restoration:
a Survey \[В Интернете\] // ResearchGate. - ResearchGate, 21 ноябрь 2021
г.. - 15 01 2022 г.. -
https://www.researchgate.net/publication/356456267_Capitalization_and_Punctuation_Restoration_a\_Survey.

Машинное обучение \[В Интернете\] // Википедия. Свободная
энциклопедия.. - 23 январь 2022 г.. - 13 февраль 2022 г.. -
https://ru.wikipedia.org/wiki/Машинное_обучение.
