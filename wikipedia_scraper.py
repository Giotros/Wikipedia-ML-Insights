import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
import random


class WikipediaHarvester:
    """Συλλογή URLs από Wikipedia category με pagination"""
    
    def __init__(self, headless: bool = True, delay_min: float = 1.0, delay_max: float = 2.0):
        """
        Args:
            headless: Αν True, ο browser τρέχει χωρίς GUI
            delay_min: Ελάχιστο delay μεταξύ requests (δευτερόλεπτα)
            delay_max: Μέγιστο delay μεταξύ requests
        """
        self.headless = headless
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.urls_collected = set()
        self.driver = None
        
    def setup_driver(self):
        """Δημιουργία Chrome WebDriver με explicit waits"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        
    def close_driver(self):
        """Κλείσιμο browser"""
        if self.driver:
            self.driver.quit()
            
    def polite_delay(self):
        """Ευγενικό delay με μικρή τυχαιότητα"""
        time.sleep(random.uniform(self.delay_min, self.delay_max))
        
    def harvest_urls(self, seed_url: str, target_count: int = 250) -> List[Dict[str, str]]:
        """
        Συλλογή URLs από Wikipedia category με pagination
        
        Args:
            seed_url: Το αρχικό URL της κατηγορίας
            target_count: Πλήθος URLs προς συλλογή
            
        Returns:
            Λίστα από dictionaries με 'title' και 'url'
        """
        self.setup_driver()
        articles = []
        current_url = seed_url
        
        try:
            while len(self.urls_collected) < target_count:
                print(f"Φόρτωση σελίδας: {current_url}")
                print(f"Συλλεγμένα URLs μέχρι τώρα: {len(self.urls_collected)}/{target_count}")
                
                self.driver.get(current_url)
                
                # Explicit wait για το container των άρθρων
                try:
                    wait = WebDriverWait(self.driver, 10)
                    wait.until(EC.presence_of_element_located((By.ID, "mw-pages")))
                    
                    # Πρόσθετο wait: περιμένουμε να φορτώσουν τα links
                    wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#mw-pages .mw-category-group ul li a")
                    ))
                    
                    # Μικρή επιπλέον αναμονή για πλήρη rendering
                    time.sleep(0.5)
                    
                except TimeoutException:
                    print("Timeout: Το περιεχόμενο δεν φόρτωσε εγκαίρως")
                    break
                
                # Εξαγωγή links από την τρέχουσα σελίδα
                article_links = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    "#mw-pages .mw-category-group ul li a"
                )
                
                for link in article_links:
                    if len(self.urls_collected) >= target_count:
                        break
                        
                    try:
                        title = link.text
                        url = link.get_attribute('href')
                        
                        if url and url not in self.urls_collected:
                            self.urls_collected.add(url)
                            articles.append({
                                'title': title,
                                'url': url
                            })
                            print(f"  ✓ {len(self.urls_collected)}: {title}")
                    except Exception as e:
                        print(f"  ✗ Σφάλμα εξαγωγής link: {e}")
                        continue
                
                # Έλεγχος για "next page"
                if len(self.urls_collected) >= target_count:
                    break
                    
                try:
                    next_link = self.driver.find_element(
                        By.LINK_TEXT, "next page"
                    )
                    current_url = next_link.get_attribute('href')
                    self.polite_delay()  # Ευγενικό delay πριν την επόμενη σελίδα
                except NoSuchElementException:
                    print("Δεν βρέθηκε 'next page' - τέλος pagination")
                    break
                    
        finally:
            self.close_driver()
            
        return articles[:target_count]


class WikipediaArticleScraper:
    """Εξαγωγή χαρακτηριστικών από Wikipedia άρθρα"""
    
    def __init__(self, headless: bool = True, delay_min: float = 1.0, delay_max: float = 2.0,
                 strict_waits: bool = True):
        """
        Args:
            headless: Browser mode
            delay_min: Minimum delay between requests
            delay_max: Maximum delay between requests
            strict_waits: Αν True, περιμένει για content rendering (πειραματισμός 1)
        """
        self.headless = headless
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.strict_waits = strict_waits
        self.driver = None
        self.stats = {
            'total': 0,
            'success': 0,
            'failures': 0,
            'timeouts': 0,
            'empty_fields': 0
        }
        
    def setup_driver(self):
        """Δημιουργία Chrome WebDriver"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(30)
        
    def close_driver(self):
        """Κλείσιμο browser"""
        if self.driver:
            self.driver.quit()
            
    def polite_delay(self):
        """Delay με τυχαιότητα"""
        time.sleep(random.uniform(self.delay_min, self.delay_max))
        
    def scrape_article(self, url: str, title: str) -> Optional[Dict]:
        """
        Εξαγωγή χαρακτηριστικών από ένα άρθρο
        
        Features που εξάγονται:
        - title: Τίτλος άρθρου
        - url: URL
        - url_hash: Hash του URL (για unique index)
        - first_paragraph: Πρώτη παράγραφος
        - page_length_chars: Μήκος περιεχομένου σε χαρακτήρες
        - num_sections: Πλήθος ενοτήτων (headings)
        - num_references: Πλήθος παραπομπών
        - has_infobox: Boolean για infobox
        - num_infobox_rows: Αριθμός γραμμών infobox
        - num_categories: Πλήθος κατηγοριών
        - num_images: Πλήθος εικόνων
        - scraped_at: Timestamp συλλογής
        """
        self.stats['total'] += 1
        
        try:
            self.driver.get(url)
            
            # Explicit wait για το main content
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.ID, "content")))
            
            if self.strict_waits:
                # Αυστηρότερο wait: περιμένουμε για παραγράφους
                try:
                    wait.until(EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "#mw-content-text p")
                    ))
                    time.sleep(0.3)  # Μικρή επιπλέον αναμονή
                except TimeoutException:
                    pass
            
            # Εξαγωγή χαρακτηριστικών
            features = {
                'title': title,
                'url': url,
                'url_hash': hashlib.md5(url.encode()).hexdigest(),
                'scraped_at': datetime.now().isoformat()
            }
            
            # First paragraph
            try:
                first_p = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    "#mw-content-text .mw-parser-output > p"
                )
                features['first_paragraph'] = first_p.text[:500]  # Πρώτα 500 chars
            except NoSuchElementException:
                features['first_paragraph'] = ""
                self.stats['empty_fields'] += 1
            
            # Page length
            try:
                content = self.driver.find_element(By.ID, "mw-content-text")
                features['page_length_chars'] = len(content.text)
            except:
                features['page_length_chars'] = 0
            
            # Number of sections (headings)
            headings = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "#mw-content-text h2, #mw-content-text h3"
            )
            features['num_sections'] = len(headings)
            
            # Number of references
            try:
                references = self.driver.find_elements(
                    By.CSS_SELECTOR, 
                    ".references li"
                )
                features['num_references'] = len(references)
            except:
                features['num_references'] = 0
            
            # Infobox
            try:
                infobox = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    "table.infobox"
                )
                features['has_infobox'] = True
                rows = infobox.find_elements(By.TAG_NAME, "tr")
                features['num_infobox_rows'] = len(rows)
            except NoSuchElementException:
                features['has_infobox'] = False
                features['num_infobox_rows'] = 0
            
            # Categories
            categories = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "#mw-normal-catlinks ul li"
            )
            features['num_categories'] = len(categories)
            
            # Images
            images = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "#mw-content-text img"
            )
            features['num_images'] = len(images)
            
            self.stats['success'] += 1
            return features
            
        except TimeoutException:
            self.stats['timeouts'] += 1
            self.stats['failures'] += 1
            print(f"  ✗ Timeout: {url}")
            return None
        except Exception as e:
            self.stats['failures'] += 1
            print(f"  ✗ Σφάλμα: {url} - {e}")
            return None
    
    def scrape_articles(self, articles: List[Dict[str, str]]) -> List[Dict]:
        """
        Scraping πολλαπλών άρθρων
        
        Args:
            articles: Λίστα με {'title', 'url'}
            
        Returns:
            Λίστα με extracted features
        """
        self.setup_driver()
        results = []
        
        try:
            for i, article in enumerate(articles, 1):
                print(f"\n[{i}/{len(articles)}] Scraping: {article['title']}")
                
                features = self.scrape_article(article['url'], article['title'])
                if features:
                    results.append(features)
                    print(f"  ✓ Επιτυχία - {features['page_length_chars']} chars, "
                          f"{features['num_sections']} sections")
                
                # Polite delay μεταξύ requests
                if i < len(articles):
                    self.polite_delay()
                    
        finally:
            self.close_driver()
            
        return results
    
    def print_stats(self):
        """Εκτύπωση στατιστικών"""
        print("\n" + "="*60)
        print("ΣΤΑΤΙΣΤΙΚΑ SCRAPING")
        print("="*60)
        print(f"Σύνολο προσπαθειών: {self.stats['total']}")
        print(f"Επιτυχίες: {self.stats['success']} ({self.stats['success']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"Αποτυχίες: {self.stats['failures']} ({self.stats['failures']/max(1,self.stats['total'])*100:.1f}%)")
        print(f"Timeouts: {self.stats['timeouts']}")
        print(f"Κενά πεδία: {self.stats['empty_fields']}")
        print("="*60)


class MongoDBHandler:
    """Διαχείριση MongoDB για αποθήκευση και deduplication"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/",
                 db_name: str = "wikipedia_ml", use_upsert: bool = True):
        """
        Args:
            connection_string: MongoDB connection string
            db_name: Όνομα database
            use_upsert: Αν True, χρήση upsert αντί για insert (πειραματισμός 4)
        """
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.use_upsert = use_upsert
        
        # Collections
        self.raw_collection = self.db['raw_articles']
        self.features_collection = self.db['clean_features']
        
        # Δημιουργία unique index στο url_hash (αν χρησιμοποιούμε upsert)
        if self.use_upsert:
            self.features_collection.create_index(
                [("url_hash", ASCENDING)], 
                unique=True
            )
    
    def save_urls(self, articles: List[Dict[str, str]]):
        """Αποθήκευση URLs στη raw collection"""
        if articles:
            self.raw_collection.insert_many(articles)
            print(f"Αποθηκεύτηκαν {len(articles)} URLs στο MongoDB")
    
    def save_features(self, features_list: List[Dict]):
        """
        Αποθήκευση features με deduplication
        
        Args:
            features_list: Λίστα από feature dictionaries
        """
        saved = 0
        updated = 0
        errors = 0
        
        for features in features_list:
            try:
                if self.use_upsert:
                    # Upsert: update αν υπάρχει, insert αν δεν υπάρχει
                    result = self.features_collection.update_one(
                        {'url_hash': features['url_hash']},
                        {'$set': features},
                        upsert=True
                    )
                    if result.upserted_id:
                        saved += 1
                    else:
                        updated += 1
                else:
                    # Απλό insert (θα δημιουργήσει duplicates)
                    self.features_collection.insert_one(features)
                    saved += 1
                    
            except DuplicateKeyError:
                errors += 1
                continue
            except Exception as e:
                print(f"Σφάλμα αποθήκευσης: {e}")
                errors += 1
                
        print(f"\nMongoDB: {saved} νέα, {updated} ενημερώσεις, {errors} σφάλματα")
    
    def export_to_jsonl(self, output_file: str = "wikipedia_features.jsonl"):
        """
        Export features σε JSONL αρχείο
        
        Args:
            output_file: Όνομα αρχείου εξόδου
        """
        features = list(self.features_collection.find({}, {'_id': 0}))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for feature in features:
                f.write(json.dumps(feature, ensure_ascii=False) + '\n')
                
        print(f"Exported {len(features)} εγγραφές στο {output_file}")
        return len(features)
    
    def get_collection_stats(self):
        """Στατιστικά από MongoDB"""
        raw_count = self.raw_collection.count_documents({})
        features_count = self.features_collection.count_documents({})
        
        print(f"\nMongoDB Stats:")
        print(f"  Raw URLs: {raw_count}")
        print(f"  Clean Features: {features_count}")
        
        return raw_count, features_count
    
    def close(self):
        """Κλείσιμο σύνδεσης"""
        self.client.close()


def main_pipeline(target_urls: int = 250, headless: bool = True, 
                   delay_min: float = 1.0, delay_max: float = 2.0,
                   strict_waits: bool = True, use_upsert: bool = True):
    """
    Πλήρης pipeline: URL harvesting → Scraping → MongoDB → Export
    
    Args:
        target_urls: Πλήθος URLs προς συλλογή
        headless: Headless browser mode
        delay_min: Minimum delay
        delay_max: Maximum delay
        strict_waits: Strict wait strategy (πειραματισμός 1)
        use_upsert: Use upsert in MongoDB (πειραματισμός 4)
    """
    seed_url = ("https://en.wikipedia.org/w/index.php?title=Category:Machine_learning"
                "&pageuntil=Probability+matching#mw-pages")
    
    print("="*60)
    print("WIKIPEDIA ML SCRAPING PIPELINE")
    print("="*60)
    print(f"Target: {target_urls} URLs")
    print(f"Headless: {headless}")
    print(f"Delay range: {delay_min}-{delay_max}s")
    print(f"Strict waits: {strict_waits}")
    print(f"Use upsert: {use_upsert}")
    print("="*60)
    
    # Βήμα 1-2: URL Harvesting
    print("\n[ΒΗΜΑ 1-2] URL Harvesting...")
    harvester = WikipediaHarvester(headless=headless, delay_min=delay_min, delay_max=delay_max)
    articles = harvester.harvest_urls(seed_url, target_urls)
    
    # Αποθήκευση URLs σε JSON
    with open('wikipedia_urls.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Αποθηκεύτηκαν {len(articles)} URLs στο wikipedia_urls.json")
    
    # Βήμα 3: Scraping χαρακτηριστικών
    print("\n[ΒΗΜΑ 3] Scraping χαρακτηριστικών...")
    scraper = WikipediaArticleScraper(
        headless=headless, 
        delay_min=delay_min, 
        delay_max=delay_max,
        strict_waits=strict_waits
    )
    features = scraper.scrape_articles(articles)
    scraper.print_stats()
    
    # Βήμα 4: MongoDB αποθήκευση
    print("\n[ΒΗΜΑ 4] Αποθήκευση στο MongoDB...")
    mongo = MongoDBHandler(use_upsert=use_upsert)
    mongo.save_urls(articles)
    mongo.save_features(features)
    mongo.get_collection_stats()
    
    # Βήμα 5: Export σε JSONL
    print("\n[ΒΗΜΑ 5] Export σε JSONL...")
    mongo.export_to_jsonl('wikipedia_features.jsonl')
    mongo.close()
    
    print("\n✓ Pipeline ολοκληρώθηκε επιτυχώς!")


if __name__ == "__main__":
    # Εκτέλεση με default παραμέτρους
    main_pipeline(
        target_urls=250,
        headless=True,
        delay_min=1.0,
        delay_max=2.0,
        strict_waits=True,
        use_upsert=True
    )