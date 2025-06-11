import requests
from bs4 import BeautifulSoup
import json
import os
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime
import logging

class BatchScraper:
    def __init__(self, base_url="https://www.deeplearning.ai/the-batch/"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_articles(self, max_articles=400):
        """Scrape articles from The Batch"""
        all_articles = []
        issue_links = self.get_issue_links()

        for issue_url in issue_links:
            articles = self.scrape_single_article(issue_url)
            for article in articles:
                if len(all_articles) >= max_articles:
                    return all_articles
                all_articles.append(article)
            time.sleep(1)

        return all_articles

    def get_issue_links(self):
        """Get all issue URLs"""
        try:
            response = self.session.get(self.base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select("a[href*='/the-batch/']")
            issue_links = set()

            for link in links:
                href = link.get('href')
                if href.startswith('/the-batch/') and href.count('/') > 2:
                    full_url = urljoin("https://www.deeplearning.ai", href.rstrip('/'))
                    issue_links.add(full_url)

            return list(issue_links)
        except Exception as e:
            logging.error(f"Error retrieving issue links: {e}")
            return []

    def scrape_single_article(self, url):
        """Scrape all articles from a single The Batch issue page"""
        articles = []
        try:
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            content_blocks = soup.find_all(['h2', 'h3', 'p', 'img'])
            current_article = {}
            article_started = False

            last_img = None  # Зберігаємо останнє зображення перед заголовком

            for tag in content_blocks:
                if tag.name == 'img':
                    img_src = tag.get('src')
                    if img_src and not img_src.startswith("data:"):
                        last_img = {
                            'url': urljoin(url, img_src),
                            'alt': tag.get('alt', ''),
                            'caption': self.extract_image_caption(tag)
                        }

                elif tag.name in ['h1', 'h2']:
                    if article_started and current_article:
                        current_article['url'] = url
                        current_article['publication_date'] = self.extract_date(soup) or str(datetime.now())
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
                current_article['publication_date'] = self.extract_date(soup) or str(datetime.now())
                current_article['scraped_at'] = str(datetime.now())
                articles.append(current_article)

            return articles

        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")
            return []


    def extract_image_caption(self, img_tag):
        """Extract image caption from surrounding elements"""
        parent = img_tag.parent
        caption_elem = parent.find('figcaption') or parent.find_next('p', class_='caption')
        return caption_elem.get_text(strip=True) if caption_elem else ""

    def extract_date(self, soup):
        """Extract publication date from soup"""
        date_elem = soup.find('time')
        return date_elem.get('datetime') if date_elem else None

    def download_images(self, articles, img_dir='data/images'):
        """Download images from articles"""
        os.makedirs(img_dir, exist_ok=True)

        for article in articles:
            for i, img_data in enumerate(article['images']):
                try:
                    response = self.session.get(img_data['url'])
                    if response.status_code == 200:
                        parsed_url = urlparse(img_data['url'])
                        filename = f"{article['title'][:50]}_{i}_{os.path.basename(parsed_url.path)}"
                        filename = "".join(c for c in filename if c.isalnum() or c in '.-_')
                        filepath = os.path.join(img_dir, filename)
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        img_data['local_path'] = filepath
                except Exception as e:
                    logging.error(f"Error downloading image {img_data['url']}: {e}")
