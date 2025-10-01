# Based on code by Andrew Niu

import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import math
import concurrent.futures
from PIL import ImageDraw, ImageFont
import time
import csv
import sys

load_dotenv()
api_key = os.getenv("API_KEY")

class GridConfig:
    def __init__(self, grid_width, grid_height, image_size, padding_percent, title_padding_percent, period, logo_path, title_font_path, text_font_path, title_font_size, text_font_size):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.image_size = image_size
        self.padding_percent = padding_percent
        self.title_height = title_font_size+int(title_font_size*2*(title_padding_percent/100))
        self.period = period
        self.logo_path = logo_path
        self.title_font_path = title_font_path
        self.text_font_path = text_font_path
        self.title_font_size = title_font_size
        self.text_font_size = text_font_size

        self.img_width, self.img_height = image_size
        self.padding = int(self.img_width * padding_percent / 100)
        self.padded_img_width = self.img_width + self.padding
        self.padded_img_height = self.img_height + self.padding

        self.title_font = ImageFont.truetype(self.title_font_path, self.title_font_size)
        self.text_font = ImageFont.truetype(self.text_font_path, self.text_font_size)

def read_users_from_csv(filepath):
    users = set()  
    with open(filepath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            users.add(row[1])  # Add the username from the second column
    return list(users)  

def fetch_scrobbles_and_distribution(user, timestamp, query_limit, period):
    total_scrobbles_url = f'http://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks&user={user}&api_key={api_key}&from={timestamp}&format=json&limit=1'
    album_distribution_url = f'http://ws.audioscrobbler.com/2.0/?method=user.getTopAlbums&user={user}&api_key={api_key}&period={period}&format=json&limit={query_limit}'

    total_scrobbles_response = requests.get(total_scrobbles_url)
    total_scrobbles_res = total_scrobbles_response.json()
    
    if "recenttracks" not in total_scrobbles_res:
        print(f"No recent tracks found for user: {user}")
        return user, 0, np.zeros(query_limit)
    
    total_scrobbles = int(total_scrobbles_res["recenttracks"]["@attr"]["totalPages"])
    
    album_distribution_response = requests.get(album_distribution_url)
    album_distribution_res = album_distribution_response.json()
    
    if "topalbums" not in album_distribution_res:
        print(album_distribution_res)
        print(user)
        return user, total_scrobbles, np.zeros(query_limit)
    
    albums = album_distribution_res['topalbums']['album']
    
    if total_scrobbles == 0:
        print(f"Total scrobbles is zero for user: {user}")
        return user, total_scrobbles, np.zeros(query_limit)
    
    distribution = [int(album["playcount"]) / total_scrobbles for album in albums]
    distribution = np.pad(distribution, (0, query_limit - len(distribution)), 'constant', constant_values=(0, 0))
    
    return user, total_scrobbles, np.array(distribution)

def fetch_scrobbles_and_distributions_concurrently(users, timestamp, query_limit, period):
    scrobble_counts = {}
    distributions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda user: fetch_scrobbles_and_distribution(user, timestamp, query_limit, period), users))
        for user, total_scrobbles, distribution in results:
            scrobble_counts[user] = total_scrobbles
            distributions.append(distribution)
    return scrobble_counts, distributions

def get_user_album_score(user, avg_album_distribution, query_limit, period):
    url = f'http://ws.audioscrobbler.com/2.0/?method=user.getTopAlbums&user={user}&api_key={api_key}&period={period}&format=json&limit={query_limit}'
    response = requests.get(url, headers={"Content-Type": "application/json"})
    res = response.json()
    
    if "topalbums" not in res:
        print(f"No top albums found for user: {user}")
        return {}
    
    albums = res["topalbums"]["album"]
    albums = [(album["name"], album["artist"]["name"], album["image"][-1]["#text"]) for album in albums if album["image"][-1]["#text"]]
    score = {album: avg_album_distribution[idx] for idx, album in enumerate(albums)}
    
    return score

def get_top_albums(users, avg_album_distribution, query_limit, period):
    album_counts = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        user_album_scores = list(executor.map(lambda user: get_user_album_score(user, avg_album_distribution, query_limit, period), users))
    
    for albums in user_album_scores:
        for album in albums.keys():
            if album in album_counts.keys():
                album_counts[album] += albums[album]
            else:
                album_counts[album] = albums[album]
    
    album_counts = {k: v for k, v in sorted(album_counts.items(), key=lambda item: item[1], reverse=True)}
    return album_counts

def download_image(url, cache_dir, retries=3, delay=2):
    filename = os.path.join(cache_dir, os.path.basename(url))
    
    # Check if the image is already in the cache
    if os.path.exists(filename):
        return Image.open(filename)

    for i in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            img = Image.open(BytesIO(response.content))
            img.save(filename)  # Save the image to the cache
            print(f"Downloaded new image: {filename}")
            return img
        except (requests.exceptions.RequestException, Image.UnidentifiedImageError) as e:
            print(f"Error downloading image from {url}: {e}")
            if i < retries - 1:
                print(f"Retrying... ({i + 1}/{retries})")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this image.")
                return None

def download_images_concurrently(image_urls, cache_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(lambda url: download_image(url, cache_dir), image_urls))
    return [img for img in images if img is not None]

def draw_title(draw, grid_img, title, logo_img, config):
    title_padding = config.title_height // 8
    title_bbox = draw.textbbox((0, 0), title, font=config.title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height_text = config.title_font.size
    logo_y = (config.title_height - logo_img.height) // 2
    title_x = (grid_img.width - title_width) // 2 + logo_img.width // 2
    title_y = (config.title_height - title_height_text) // 2 - title_padding
    draw.text((title_x, title_y), title, fill=(255, 255, 255), font=config.title_font)
    grid_img.paste(logo_img, (title_x - logo_img.width - title_padding, logo_y))

def calculate_text_area_width(image_info, text_font):
    max_text_width = 0
    for artist, album, _ in image_info:
        text_bbox = text_font.getbbox(f"{artist} - {album}")
        text_width = text_bbox[2] - text_bbox[0]
        if text_width > max_text_width:
            max_text_width = text_width
    return max_text_width

def calculate_total_width(config, text_area_width):
    return config.grid_width * config.padded_img_width + config.padding + text_area_width + config.padding

def calculate_text_sizing(config, image_info, text_spacing):
    num_images = len(image_info)
    text_font = config.text_font
    text_font_size = config.text_font_size
    
    for index in range(0, num_images, config.grid_width):
        row_text_height = 0
        for i in range(config.grid_width):
            if index + i < num_images:
                row_text_height += text_font_size + text_spacing
        row_text_height -= text_spacing

        if row_text_height > config.padded_img_width:
            shrink_factor = config.padded_img_width / row_text_height
            text_font_size = int(config.text_font_size * shrink_factor)
            text_spacing = int(text_spacing * shrink_factor)
            text_font = ImageFont.truetype(config.text_font_path, text_font_size)
    
    return text_font, text_font_size, text_spacing

def draw_images_and_text(draw, grid_img, images, image_info, text_font, text_font_size, text_spacing, config):
    num_images = len(image_info)
    for index, (artist, album, _) in enumerate(image_info):
        img = images[index].resize((config.padded_img_width - config.padding, config.padded_img_height - config.padding))
        x = (index % config.grid_width) * config.padded_img_width + config.padding
        y = (index // config.grid_width) * config.padded_img_height + config.padding + config.title_height
        grid_img.paste(img, (x, y))

        if index % config.grid_width == 0:
            draw_text(draw, text_font, text_font_size, text_spacing, config, y, image_info, index, num_images)

def draw_text(draw, text_font, text_font_size, text_spacing, config, y, image_info, index, num_images):
    text_x = config.grid_width * config.padded_img_width + config.padding
    font_top_offset = 5
    text_y = y - text_font_size // font_top_offset
    row_texts = []
    for i in range(config.grid_width):
        if index + i < num_images:
            artist, album, _ = image_info[index + i]
            row_texts.append((artist, album))

    for artist, album in row_texts:
        draw.text((text_x, text_y), f"{artist} - {album}", fill=(255, 255, 255), font=text_font)
        text_y += text_font.size + text_spacing

def create_image_grid(image_info, config, cache_dir):
    period_display_titles = {
        "1week": "weekly",
        "1month": "monthly",
        "1year": "yearly"
    }
    title = f"indieheads {period_display_titles[config.period]} chart"

    logo_size = int(config.title_font_size * 0.875)
    logo_img = Image.open(config.logo_path).resize((logo_size, logo_size))
    images = download_images_concurrently([info[2] for info in image_info], cache_dir)
    text_spacing = config.text_font.size // 10

    text_font, text_font_size, text_spacing = calculate_text_sizing(config, image_info, text_spacing)
    text_area_width = calculate_text_area_width(image_info, text_font)
    total_width = calculate_total_width(config, text_area_width)
    total_height = config.grid_height * config.padded_img_height + config.padding + config.title_height
    grid_img = Image.new('RGB', (total_width, total_height), color=(0, 0, 0))

    draw = ImageDraw.Draw(grid_img)
    draw_title(draw, grid_img, title, logo_img, config)

    draw_images_and_text(draw, grid_img, images, image_info, text_font, text_font_size, text_spacing, config)

    return grid_img

def print_dict(d, config):
    count = 0
    image_info = []
    for k, v in d.items():
        if count == config.grid_width * config.grid_height:
            break
        print(f"{k[1]} - {k[0]}: {v}")
        image_info.append((k[1], k[0], k[2]))  # Append the artist, album, and image URL
        count += 1
    return image_info

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python grouptopster.py <period>")
        print("Available periods: week, month, year")
        sys.exit(1)
    
    period_arg = sys.argv[1].lower()
    
    # Map user-friendly names to API period codes
    period_mapping = {
        "week": "1week",
        "month": "1month", 
        "year": "1year"
    }
    
    if period_arg not in period_mapping:
        print(f"Invalid period: {period_arg}")
        print("Available periods: week, month, year")
        sys.exit(1)
    
    selected_period = period_mapping[period_arg]
    
    users = read_users_from_csv("Indieheads topster chart.csv")
    
    # Configuration
    config = GridConfig(
        grid_width=10,  # Number of albums per row
        grid_height=10,  # Number of albums per column
        image_size=(300, 300),
        padding_percent=7,
        title_padding_percent=12.5,
        period=selected_period,  # Use the selected period from command line
        logo_path="logo.png",
        title_font_path="GeographicaHand.ttf",
        text_font_path="Monospace.ttf",
        title_font_size=80,
        text_font_size=25
    )

    query_limit = 500  #limit the number of albums fetched

    period_options = {
        "1week": 7,
        "1month": 30,
        "1year": 365
    }
    days = period_options[config.period]
    timeago = datetime.now() - timedelta(days)
    timestamp = int((timeago - datetime(1970, 1, 1)).total_seconds())

    scrobble_counts, distributions = fetch_scrobbles_and_distributions_concurrently(users, timestamp, query_limit, config.period)

    avg_album_distribution = np.mean(distributions, axis=0)

    top_albums = get_top_albums(users, avg_album_distribution, query_limit, config.period)
    image_info = print_dict(top_albums, config)

    cache_dir = "./cached_art"
    os.makedirs(cache_dir, exist_ok=True)

    grid_img = create_image_grid(image_info, config, cache_dir)
    output_path = "final_image.png"
    grid_img.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    main()