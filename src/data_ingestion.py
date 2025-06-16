import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime
import logging
import re

class BatchScraper:
    def __init__(self, base_url="https://www.deeplearning.ai/the-batch/"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
            })

    def get_all_issue_urls(self):
        issue_links = set()
        links_on_links=set()
        page_num = 1
        
        while True:
            if page_num == 1:
                url = "https://www.deeplearning.ai/the-batch/"  # перша сторінка без /page/1/
            else:
                url = f"{self.base_url}/page/{page_num}/"

            try:
                response = self.session.get(url)
                if response.status_code == 404:
                    print("Page not found, stopping scraping."+url)
                    break  # закінчити цикл, якщо сторінка не знайдена

                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.select("a[href*='/the-batch/']")

                for link in links:
                    href = link.get('href')
                    if href.startswith('/the-batch/') and href.count('/') > 2 and "the-batch/page/" not in href :
                        if "/tag/" in href:
                            links_on_links.add(urljoin("https://www.deeplearning.ai", href.rstrip('/')))
                        else:
                            full_url = urljoin("https://www.deeplearning.ai", href.rstrip('/'))
                            issue_links.add(full_url)

                page_num += 1
                time.sleep(1)  # пауза в 1 секунду між запитами

            except Exception as e:
                logging.error(f"Error retrieving issue links from {url}: {e}")
                break
        for url in links_on_links:
            try:
                response = self.session.get(url)
                if response.status_code != 200:
                    print(f"Page not found, skipping {url}")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                links = soup.select("a[href*='/the-batch/']")

                for link in links:
                    href = link.get('href')
                    if href.startswith('/the-batch/') and href.count('/') > 2 and "the-batch/page/" not in href and "/tag/" not in href:
                        full_url = urljoin("https://www.deeplearning.ai", href.rstrip('/'))
                        issue_links.add(full_url)

                time.sleep(1)

            except Exception as e:
                logging.error(f"Error processing links_on_links url {url}: {e}")

        json.dump(list(issue_links), open("issue_links.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        json.dump(list(links_on_links), open("links_on_links.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)
        return list(issue_links)
    
    def scrape_issue_article(self, url):
        """Scrape all articles from a single The Batch issue page"""
        
        articles = []
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            content_blocks = soup.find_all(['h1',"h2",'p', 'img'])
            current_article = {}
            article_started = False
            found_news_heading = False

            last_img = None  # Зберігаємо останнє зображення перед заголовком

            for tag in content_blocks:
                if (tag.name == 'h2' or tag.name =="h1" ) and "news" in tag.get_text(strip=True).lower():
                    found_news_heading = True  # 👈 дозволяємо почати обробку після цього
                    continue  # пропускаємо сам заголовок "news"
                elif not found_news_heading:
                    continue  # ⛔ пропускаємо все до заголовка "news"
                if tag.name == 'img':
                    img_src = tag.get('src')
                    if img_src == "/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fdlai-batch-logo.a60dbb9f.png&w=640&q=75":
                        # Skip this article by breaking out of the loop
                        articles = []
                        continue
                    if img_src and not img_src.startswith("data:"):
                        last_img = {
                            'url': urljoin(url, img_src),
                            'alt': tag.get('alt', ''),
                            'caption': self.extract_image_caption(tag)
                        }

                elif tag.name in ['h2', 'h1'] :
                    if article_started and current_article:
                        current_article['url'] = url
                        pub_date_elem = soup.find('div', class_='mt-1 text-slate-600 text-base text-sm')
                        print(f"pub_date_elem: {pub_date_elem}")
                        if pub_date_elem and pub_date_elem.get_text(strip=True):
                            current_article['publication_date'] = pub_date_elem.get_text(strip=True)
                        else:
                            current_article['publication_date'] = str(datetime.now())
                        current_article['scraped_at'] = str(datetime.now())
                        articles.append(current_article)
                        current_article = {}

                    current_article['title'] = tag.get_text(strip=True)
                    current_article['content'] = ""
                    current_article['images'] = []

                    if last_img:
                        current_article['images'].append(last_img)
                        last_img = None  # Очистити, щоб не дублювати в наступній статті

                    article_started = True

                elif tag.name == 'p' and article_started:
                    current_article['content'] += tag.get_text(strip=True) + " "

                elif tag.name == 'img' and article_started:
                    img_src = tag.get('src')
                    if img_src and not img_src.startswith("data:"):
                        img_url = urljoin(url, img_src)
                        current_article['images'].append({
                            'url': img_url,
                            'alt': tag.get('alt', ''),
                            'caption': self.extract_image_caption(tag)
                        })

            if current_article and article_started:
                current_article['url'] = url
                pub_date_elem = soup.find('div', class_='mt-1 text-slate-600 text-base text-sm')
                current_article['publication_date'] = pub_date_elem.get_text(strip=True) or str(datetime.now())
                current_article['scraped_at'] = str(datetime.now())
                articles.append(current_article)

            return articles

        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return []

    def scrape_simple_article(self, url):
        """Scrape a single The Batch article page (title, date, paragraphs, main image)"""
        article = {}

        try:
            response = self.session.get(url)
            if response.status_code != 200:
                logging.error(f"Failed to fetch {url}: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            # Заголовок
            title_tag = soup.find('h1')
            article['title'] = title_tag.get_text(strip=True) if title_tag else 'No Title'

            # Публікація
            date_tags = soup.find_all('div', class_="inline-flex px-3 py-1 text-sm font-normal transition-colors rounded-md bg-slate-200 hover:bg-slate-300 text-slate-500")
            for tag in date_tags:
                text = tag.get_text(strip=True)
                if re.search(r'\d', text):
                    article['publication_date'] = text
                    break  # зупинимося на першому валідному
                else:
                    article['publication_date'] = None

            # Основне зображення
            main_img_tags = soup.find_all('img', attrs={'alt': True, 'srcset': True})
            for main_img_tag in main_img_tags:
                if "batch-logo."  not in main_img_tag.get('srcset', '') :
                    srcset = main_img_tag.get('srcset', '')
                    
                    # Вибираємо останнє зображення з найвищою роздільністю
                    last_img = srcset.split(',')[-1].strip().split(' ')[0]
                    article['image'] = {
                        'url': urljoin(url, last_img),
                        'alt': main_img_tag.get('alt', '')
                    }
                    break
                else:
                    article['image'] = None

            # Контент
            paragraphs = soup.find_all('p')
            content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            article['content'] = content

            # Метадані
            article['url'] = url
            article['scraped_at'] = str(datetime.now())

            # Зберегти у файл
            with open("articles2.json", "w", encoding="utf-8") as f:
                json.dump([article], f, ensure_ascii=False, indent=4)

            return [article]

        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return []

        """Extract publication date from soup"""
        date_elem = soup.find('time')
        return date_elem.get('datetime') if date_elem else None

    def download_articles(self, output_dir='data/articles'):
        """Download articles to JSON files"""
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Fix path joining with proper separator
        scraped_articles_path = os.path.join(output_dir, "scraped_articles.json")
        issue_links_path = os.path.join(output_dir, "issue_links.json")

        try:
            with open(scraped_articles_path, "r", encoding="utf-8") as f:
                scraped_articles = json.load(f)
                return scraped_articles
        except FileNotFoundError:
            print("No previously scraped articles found. Starting fresh.")
            try:
                with open(issue_links_path, "r", encoding="utf-8") as f:
                    issue_urls = json.load(f)
            except FileNotFoundError:
                print("No issue URLs found. Scraping all issues.")
                issue_urls = self.get_all_issue_urls()
            
            results = []
            failed_urls = []
            
            for idx, url in enumerate(issue_urls, start=1):
                if idx % 100 == 0:
                    print(f"[{idx}/{len(issue_urls)}] Scraping: {url}")
                
                try:
                    if "the-batch/issue-" in url:
                        articles = self.scrape_issue_article(url)
                    else:
                        articles = self.scrape_simple_article(url)
                    results.extend(articles)

                except Exception as e:
                    logging.error(f"Error scraping {url}: {e}")
                    print(f"❌ Failed: {url}")
                    failed_urls.append(url)

                # Respectful crawling delay
                time.sleep(1.5)  # adjust to 2–3 seconds if you're hitting rate limits

            # Save all successfully scraped articles
            with open(scraped_articles_path, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            print(f"✅ Successfully scraped {len(results)} articles.")
            return results