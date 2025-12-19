"""
NAME:        src/utils/logger.py
VERSION:     1.0
DESCRIPTION: A simple logger that redirects any info being printed to 
             terminal, also to a specific .txt log file
"""

import sys
import os


class Logger(object):

    def __init__(self, filepath):
        """
        Simple logger that redirects sys.stdout to terminal and specific
        file simulataneously
            filepath (str)
                The directory of log file 
        """


        # Any information printed to terminal is sys.stdout
        self.terminal = sys.stdout

        # Check directory exists
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except FileNotFoundError:
            print("Experimental Directory doesn't exist")

        # Creat logs dir in experiment dir then open a new .txt file
        self.log_file = open(filepath, "a", encoding='utf-8')



    def write(self, message):
        # write to terminal
        self.terminal.write(message)
        # Write to file
        self.log_file.write(message)
        # Ensure it's written immediately so no loss if crash
        self.flush



    def flush(self):
        # Needed for compatability with sys.stdout interface
        self.terminal.flush()
        self.log_file.flush()



    def close(self):
        self.log_file.close()
