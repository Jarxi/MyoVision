
import subprocess

while 1:
    name = input("Name of application: ")
    if name == 'spotify':
        subprocess.call(
            ["/usr/bin/open", "-n", "-a", "/Applications/Spotify.app"]
            )
    if name == 'itunes':
            subprocess.call(
            ["/usr/bin/open", "-n", "-a", "/Applications/iTunes.app"]
            )