# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:58:58 2024

@author: Jayyy
"""

import pathlib
import textgrids
import sys

class Read_Textgrid:
    """
    A class to read and parse TextGrid files for phoneme data.
    """
    def __init__(self, path):
        """
        Initialize a Read_Textgrid instance.
        
        Parameters:
        path (str): The path to the TextGrid file.
        """
        self.pathname = pathlib.Path(path)
        # Try to open the file as textgrid
        try:
            self.grid = textgrids.TextGrid(self.pathname)
            # Print a success message
            print(f'Successfully loaded: {self.pathname.stem}')       
            
            for phone in self.grid['phones']:
                # Convert Praat to Unicode in the label
                label = phone.text.transcode()
                # Print label and phoneme timing, CSV-like
                print('"{}";{}, {}'.format(label, phone.xmin, phone.xmax))
                
        except FileNotFoundError:
            print(f'File not found: {self.pathname.stem}', file=sys.stderr)
        
        except PermissionError:
            print(f'Cannot read: {self.pathname.stem}', file=sys.stderr)
        
        except (textgrids.ParseError, textgrids.BinaryError):
            print(f'Invalid file format: {self.pathname.stem}', file=sys.stderr)
        
        except Exception as e:
            print(f'An unexpected error occurred: {e}', file=sys.stderr)

