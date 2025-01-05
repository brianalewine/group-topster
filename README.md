# GroupTopster
GroupTopster is a Python script that generates a visual grid of top albums for a group of users from the Indieheads community. It fetches data from the Last.fm API, processes the data to determine the most popular albums, and creates an image grid displaying these albums along with their respective artists.

![Example Image](/images/example_chart.png)
## Usage

Instructions on how to use the project.

1. Ensure you have the required dependencies installed. You can install them using:
    ```sh
    pip install -r requirements.txt
    ```

2. Create a .env file in the project directory and add your Last.fm API key:
    ```
    API_KEY=your_api_key_here
    ```

3. Prepare a CSV file named `Indieheads topster chart.csv` with a list of users. The CSV should have a header and at least one column containing the usernames.

4. Run the script:
    ```sh
    python grouptopster.py
    ```

5. The script will generate an image named final_image.png in the project directory, displaying the top albums for the specified period.

## Credits

This project is built upon code by Andrew Niu.