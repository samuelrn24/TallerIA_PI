import os
from huggingface_hub import InferenceClient
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate images with Hugging Face and update movie image field"

    def handle(self, *args, **kwargs):
        # ✅ Load environment variables from the .env file
        load_dotenv('../openAI.env')

        # ✅ Initialize the Hugging Face client with the API token
        client = InferenceClient(
            token=os.environ.get('HF_TOKEN')
        )
        
        # ✅ Folder to save images
        images_folder = 'media/movie/images/'
        os.makedirs(images_folder, exist_ok=True)

        # ✅ Fetch all movies
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            try:
                # ✅ Call the helper function
                image_relative_path = self.generate_and_download_image(client, movie.title, images_folder)

                # ✅ Update database
                movie.image = image_relative_path
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {e}")

            # 🔎 Process just the first movie for demonstration (Do not remove the break yet)
            #break

        self.stdout.write(self.style.SUCCESS("Process finished (only first movie updated)."))

    def generate_and_download_image(self, client, movie_title, save_folder):
        """
        Generates an image using Hugging Face's Stable Diffusion model and saves it.
        Returns the relative image path or raises an exception.
        """
        prompt = f"A high quality movie poster of {movie_title}, cinematic lighting, no text"

        # ✅ Generate image with Hugging Face (returns a PIL Image object)
        image = client.text_to_image(
            prompt=prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0" 
        )

        # ✅ Prepare the filename (cleaning the title for valid file paths) and full save path
        clean_title = "".join(c for c in movie_title if c.isalnum() or c in " ").replace(" ", "_")
        image_filename = f"m_{clean_title}.png"
        image_path_full = os.path.join(save_folder, image_filename)

        # ✅ Save the image locally
        image.save(image_path_full)

        # ✅ Return relative path to be saved in the DB
        return os.path.join('movie/images', image_filename)