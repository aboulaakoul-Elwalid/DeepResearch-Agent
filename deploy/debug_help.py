import modal

image = modal.Image.from_registry("gradientservice/parallax:latest", add_python="3.11")
app = modal.App("debug-parallax")

@app.function(image=image, gpu="L4")
def get_help():
    import subprocess
    print("--------- PARALLAX HELP MENU ---------")
    # We try both the main help and the run subcommand help
    try:
        subprocess.run(["parallax", "--help"], check=False)
        print("\n\n--------- RUN SUBCOMMAND HELP ---------")
        subprocess.run(["parallax", "run", "--help"], check=False)
    except Exception as e:
        print(e)
