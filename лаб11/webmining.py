import requests
from bs4 import BeautifulSoup
import csv

# 1. Налаштування цільового URL
url = "http://quotes.toscrape.com/"


def scrape_quotes(target_url):
    print(f"Починаємо збір даних з: {target_url}")

    # 2. Відправка GET запиту
    try:
        response = requests.get(target_url)

        # Перевірка статусу відповіді [cite: 95, 102]
        if response.status_code == 200:
            print("З'єднання успішне (Code 200)")

            # 3. Створення об'єкта BeautifulSoup для парсингу [cite: 149]
            soup = BeautifulSoup(response.content, "html.parser")

            # 4. Пошук елементів. На цьому сайті кожна цитата в блоці <div class="quote">
            quotes_blocks = soup.find_all("div", class_="quote")

            scraped_data = []

            # 5. Ітерація по знайдених блоках [cite: 152]
            for block in quotes_blocks:
                # Витягуємо текст цитати (тег <span> з класом text)
                text = block.find("span", class_="text").text.strip()

                # Витягуємо автора (тег <small> з класом author)
                author = block.find("small", class_="author").text.strip()

                # Витягуємо теги (теги <a> всередині <div class="tags">)
                tags_meta = block.find("div", class_="tags").find_all("a", class_="tag")
                tags = [tag.text for tag in tags_meta]

                # Додаємо в список
                scraped_data.append({
                    "author": author,
                    "text": text,
                    "tags": tags
                })

            return scraped_data
        else:
            print(f"Помилка доступу: {response.status_code}")
            return []

    except Exception as e:
        print(f"Виникла помилка: {e}")
        return []


# Запуск програми
data = scrape_quotes(url)

# 6. Представлення результатів [cite: 8]
print("\n Результати Web Scraping")
for i, item in enumerate(data, 1):
    print(f"{i}. Автор: {item['author']}")
    print(f"   Цитата: {item['text'][:50]}...")  # Друкуємо перші 50 символів
    print(f"   Теги: {', '.join(item['tags'])}")
    print("-" * 20)