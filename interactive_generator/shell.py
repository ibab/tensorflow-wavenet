#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from suds.client import Client
from termcolor import colored
from time import sleep
import urllib
import subprocess
import cmd, sys
import xml.etree.ElementTree

"""

This is an interactive script for testing EveNet.

TODO:
  - Map from Cereproc phonemes to Sophia phonemes
  - Integrate with ROS & Generator script.
  - Remove my personal password from this script :)
  
"""

class EveShell(cmd.Cmd):

    # Cerevoice
    client = Client("https://cerevoice.com/soap/soap_1_1.php?WSDL")

    # Emotions
    emotions = [colored('Angry', 'red'),
                colored("Happy", 'blue'),
                colored("Sad", 'grey', attrs=['bold']),
                colored("Neutral", 'white'),
                colored("Frustrated", 'yellow'),
                colored("Excited", 'cyan'),
                colored("Fearful", 'magenta'),
                colored("Disgusted", 'green'),
                "Other"]
    currentEmotion = 0

    # Config
    fps = 48.0

    # Config
    intro = '\nWelcome to EveNet interactive shell.   Type help or ? to list commands.\n'
    prompt = '(' + emotions[currentEmotion] + ') '

    # ----- basic turtle commands -----
    def do_say(self, arg):
        'Say something using current emotion.'
        speakString = arg
        request = EveShell.client.service.speakExtended("59746f2101ec1", "LJpcC67e3u", "Heather", speakString, metadata=True)

        if request.resultCode == 1:
            print(" %s" % request.resultDescription)

            urlOpener = urllib.URLopener()
            urlOpener.retrieve(request.fileUrl, "/tmp/sound.ogg")
            urlOpener.retrieve(request.metadataUrl, "/tmp/metadata.xml")



            # parse XML
            metaDataTree = xml.etree.ElementTree.parse('/tmp/metadata.xml').getroot()
            phonemeFrames = []

            for phoneme in metaDataTree.findall("phone"):
                value = phoneme.items()[2][1]
                start = float(phoneme.items()[0][1])
                end = float(phoneme.items()[1][1])
                nrOfFrames = int(round((end-start)*EveShell.fps))

                for _ in range(nrOfFrames):
                    phonemeFrames.append(value)

            # play in background...
            play("/tmp/sound.ogg")

            # Stream the phonemes
            for phoneme in phonemeFrames:
                print("Current phoneme: %s " % colored(phoneme, 'white', 'on_grey', attrs=['bold']), end=' \r ')
                sys.stdout.flush()
                sleep(1.0 / EveShell.fps)

            print("")

        else:
            print("ERROR")
            print(request)

    def do_emo(self, arg):
        'Switch emotion according to ID'
        EveShell.currentEmotion = parse(arg)[0]
    def do_angry(self, arg):
        EveShell.currentEmotion = 0
    def do_happy(self, arg):
        EveShell.currentEmotion = 1
    def do_sad(self, arg):
        EveShell.currentEmotion = 2
    def do_neutral(self, arg):
        EveShell.currentEmotion = 3
    def do_frustrated(self, arg):
        EveShell.currentEmotion = 4
    def do_excited(self, arg):
        EveShell.currentEmotion = 5
    def do_fearful(self, arg):
        EveShell.currentEmotion = 6
    def do_disgusted(self, arg):
        EveShell.currentEmotion = 7
    def do_other(self, arg):
        EveShell.currentEmotion = 8

    # ----- record and playback -----
    def postcmd(self, stop, line):
        EveShell.prompt = '(' + EveShell.emotions[EveShell.currentEmotion] + ') '
        return False
    def close(self):
        print("byebye")


def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    return tuple(map(int, arg.split()))

def play(audio_file_path):
    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-nostats", "-loglevel", "0", audio_file_path])


if __name__ == '__main__':
    try:
        shell = EveShell()

        for i,emo in enumerate(shell.emotions):
            print("%d: %s" % (i, emo))

        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nkthxbye")
