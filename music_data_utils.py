# Tools to load and save midi files for the rnn-gan-project.
# 
# Written by Olof Mogren, http://mogren.one/
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import urlparse, urllib2, os, midi, math, random, re, string
import numpy as np

sources                              = {}
sources['classical']                 = {}
#sources['classical']['misc']         = ['http://www.midiworld.com/classic.htm']
sources['classical']['alkan']        = ['http://www.classicalmidi.co.uk/alkan.htm']
sources['classical']['adam']         = ['http://www.classicalmidi.co.uk/adam.htm']
sources['classical']['aguado']       = ['http://www.classicalmidi.co.uk/aguadodion.htm']
sources['classical']['albenizisaac'] = ['http://www.classicalmidi.co.uk/albeniz.htm']
sources['classical']['albenizmateo'] = ['http://www.classicalmidi.co.uk/albenizmateo.htm']
sources['classical']['albinoni']     = ['http://www.classicalmidi.co.uk/albinoni.htm']
sources['classical']['alford']       = ['http://www.classicalmidi.co.uk/alford.htm']
sources['classical']['anderson']     = ['http://www.classicalmidi.co.uk/anderson.htm']
sources['classical']['ansell']       = ['http://www.classicalmidi.co.uk/anselljohn.htm']
sources['classical']['arensky']      = ['http://www.classicalmidi.co.uk/arensky.htm']
sources['classical']['arriaga']      = ['http://www.classicalmidi.co.uk/arriag.htm']
sources['classical']['bach']         = ['http://www.midiworld.com/bach.htm','http://www.classicalmidi.co.uk/bach.htm']
sources['classical']['bartok']       = ['http://www.midiworld.com/bartok.htm','http://www.classicalmidi.co.uk/bartok.htm']
sources['classical']['barber']       = ['http://www.classicalmidi.co.uk/barber.htm']
sources['classical']['barbieri']     = ['http://www.classicalmidi.co.uk/barbie.htm']
sources['classical']['bax']          = ['http://www.classicalmidi.co.uk/bax.htm']
sources['classical']['beethoven']    = ['http://www.midiworld.com/beethoven.htm','http://www.classicalmidi.co.uk/beethoven.htm']
sources['classical']['bellini']      = ['http://www.classicalmidi.co.uk/bellini.htm']
sources['classical']['berlin']       = ['http://www.classicalmidi.co.uk/berlin.htm']
sources['classical']['berlioz']      = ['http://www.classicalmidi.co.uk/berlioz.htm']
sources['classical']['binge']        = ['http://www.classicalmidi.co.uk/binge.htm']
sources['classical']['bizet']        = ['http://www.classicalmidi.co.uk/bizet.htm']
sources['classical']['boccherini']   = ['http://www.classicalmidi.co.uk/bocc.htm']
sources['classical']['boellman']     = ['http://www.classicalmidi.co.uk/boell.htm']
sources['classical']['borodin']      = ['http://www.classicalmidi.co.uk/borodin.htm']
sources['classical']['boyce']        = ['http://www.classicalmidi.co.uk/boyce.htm']
sources['classical']['brahms']       = ['http://www.midiworld.com/brahms.htm','http://www.classicalmidi.co.uk/brahms.htm']
sources['classical']['breton']       = ['http://www.classicalmidi.co.uk/breton.htm']
sources['classical']['britten']      = ['http://www.classicalmidi.co.uk/britten.htm']
sources['classical']['bouwer']       = ['http://www.classicalmidi.co.uk/bouwer.htm']
sources['classical']['bruch']        = ['http://www.classicalmidi.co.uk/bruch.htm']
sources['classical']['bruckner']     = ['http://www.classicalmidi.co.uk/bruck.htm']
sources['classical']['bergmuller']   = ['http://www.classicalmidi.co.uk/bergmuller.htm']
sources['classical']['busoni']       = ['http://www.classicalmidi.co.uk/busoni.htm']
sources['classical']['byrd']         = ['http://www.midiworld.com/byrd.htm','http://www.classicalmidi.co.uk/byrd.htm']
sources['classical']['carulli']      = ['http://www.classicalmidi.co.uk/carull.htm']
sources['classical']['chabrier']     = ['http://www.classicalmidi.co.uk/chabrier.htm']
sources['classical']['chaminade']    = ['http://www.classicalmidi.co.uk/chaminad.htm']
sources['classical']['chapi']        = ['http://www.classicalmidi.co.uk/chapie.htm']
sources['classical']['cherubini']    = ['http://www.classicalmidi.co.uk/cherub.htm']
sources['classical']['chopin']       = ['http://www.midiworld.com/chopin.htm','http://www.classicalmidi.co.uk/chopin.htm']
sources['classical']['clementi']     = ['http://www.classicalmidi.co.uk/clemen.htm']
sources['classical']['coates']       = ['http://www.classicalmidi.co.uk/coates.htm']
sources['classical']['copland']      = ['http://www.classicalmidi.co.uk/copland.htm']
sources['classical']['corelli']      = ['http://www.classicalmidi.co.uk/cor.htm']
sources['classical']['cramer']       = ['http://www.classicalmidi.co.uk/cramer.htm']
sources['classical']['curzon']       = ['http://www.classicalmidi.co.uk/cuzon.htm']
sources['classical']['czerny']       = ['http://www.classicalmidi.co.uk/czerny.htm']
sources['classical']['debussy']      = ['http://www.classicalmidi.co.uk/debussy.htm']
sources['classical']['delibes']      = ['http://www.classicalmidi.co.uk/del.htm']
#sources['classical']['dukas']        = ['http://www.classicalmidi.co.uk/dukas.htm']
sources['classical']['delius']       = ['http://www.classicalmidi.co.uk/delius.htm']
sources['classical']['dialoc']       = ['http://www.classicalmidi.co.uk/diaoc.htm']
sources['classical']['dupre']        = ['http://www.classicalmidi.co.uk/dupre.htm']
sources['classical']['dussek']       = ['http://www.classicalmidi.co.uk/dussek.htm']
sources['classical']['dvorak']       = ['http://www.classicalmidi.co.uk/dvok.htm']
sources['classical']['elgar']        = ['http://www.classicalmidi.co.uk/elgar.htm']
sources['classical']['eshpai']       = ['http://www.classicalmidi.co.uk/Eshpai.htm', 'http://www.classicalmidi.co.uk/Eshpai%20.htm']
sources['classical']['faure']        = ['http://www.classicalmidi.co.uk/faure.htm']
sources['classical']['field']        = ['http://www.classicalmidi.co.uk/field.htm']
sources['classical']['flotow']       = ['http://www.classicalmidi.co.uk/flotow.htm']
sources['classical']['foster']       = ['http://www.classicalmidi.co.uk/foster.htm']
sources['classical']['franck']       = ['http://www.classicalmidi.co.uk/franck.htm']
sources['classical']['fresc']        = ['http://www.classicalmidi.co.uk/fresc.htm']
sources['classical']['garoto']       = ['http://www.classicalmidi.co.uk/garoto.htm']
sources['classical']['german']       = ['http://www.classicalmidi.co.uk/german.htm']
sources['classical']['gershwin']     = ['http://www.classicalmidi.co.uk/gershwin.htm']
sources['classical']['gilbert']      = ['http://www.classicalmidi.co.uk/gilbert.htm']
sources['classical']['ginast']       = ['http://www.classicalmidi.co.uk/ginast.htm']
sources['classical']['gott']         = ['http://www.classicalmidi.co.uk/gott.htm']
sources['classical']['gounod']       = ['http://www.classicalmidi.co.uk/gounod.htm']
sources['classical']['grain']        = ['http://www.classicalmidi.co.uk/grain.htm']
sources['classical']['grieg']        = ['http://www.classicalmidi.co.uk/grieg.htm']
sources['classical']['griff']        = ['http://www.classicalmidi.co.uk/griff.htm']
sources['classical']['haydn']        = ['http://www.midiworld.com/haydn.htm','http://www.classicalmidi.co.uk/haydn.htm']
sources['classical']['handel']       = ['http://www.midiworld.com/handel.htm','http://www.classicalmidi.co.uk/handel.htm']
sources['classical']['heller']       = ['http://www.classicalmidi.co.uk/heller.htm']
sources['classical']['herold']       = ['http://www.classicalmidi.co.uk/herold.htm']
sources['classical']['hiller']       = ['http://www.classicalmidi.co.uk/hiller.htm']
sources['classical']['holst']        = ['http://www.classicalmidi.co.uk/holst.htm']
sources['classical']['hummel']       = ['http://www.midiworld.com/hummel.htm','http://www.classicalmidi.co.uk/hummel.htm']
sources['classical']['ibert']        = ['http://www.classicalmidi.co.uk/ibert.htm']
sources['classical']['ives']         = ['http://www.classicalmidi.co.uk/ives.htm']
sources['classical']['janacek']      = ['http://www.classicalmidi.co.uk/janacek.htm']
sources['classical']['joplin']       = ['http://www.classicalmidi.co.uk/joplin.htm']
sources['classical']['jstrauss']     = ['http://www.classicalmidi.co.uk/jstrauss.htm']
sources['classical']['karg']         = ['http://www.classicalmidi.co.uk/karl.htm']
sources['classical']['khach']        = ['http://www.classicalmidi.co.uk/khach.htm']
sources['classical']['kuhlau']       = ['http://www.classicalmidi.co.uk/kuhlau.htm']
sources['classical']['lalo']         = ['http://www.classicalmidi.co.uk/lalo.htm']
sources['classical']['lemire']       = ['http://www.classicalmidi.co.uk/lemire.htm']
sources['classical']['lenar']        = ['http://www.classicalmidi.co.uk/lenar.htm']
sources['classical']['liszt']        = ['http://www.midiworld.com/liszt.htm','http://www.classicalmidi.co.uk/liszt.htm']
sources['classical']['lobos']        = ['http://www.classicalmidi.co.uk/lobos.htm']
sources['classical']['lovland']      = ['http://www.classicalmidi.co.uk/lovland.htm']
sources['classical']['lyssen']       = ['http://www.classicalmidi.co.uk/lyssen.htm']
sources['classical']['maccunn']      = ['http://www.classicalmidi.co.uk/maccunn.htm']
sources['classical']['mahler']       = ['http://www.classicalmidi.co.uk/mahler.htm']
sources['classical']['maier']        = ['http://www.classicalmidi.co.uk/maier.htm']
sources['classical']['marcello']     = ['http://www.classicalmidi.co.uk/marcello.htm']
sources['classical']['martini']      = ['http://www.classicalmidi.co.uk/martini.htm']
sources['classical']['mehul']        = ['http://www.classicalmidi.co.uk/mehul.htm']
sources['classical']['mendelssohn']  = ['http://www.midiworld.com/mendelssohn.htm','http://www.classicalmidi.co.uk/mend.htm']
sources['classical']['messager']     = ['http://www.classicalmidi.co.uk/messager.htm']
sources['classical']['messia']       = ['http://www.classicalmidi.co.uk/messia.htm']
sources['classical']['meyerbeer']    = ['http://www.classicalmidi.co.uk/meyerbeer.htm']
sources['classical']['modest']       = ['http://www.classicalmidi.co.uk/modest.htm']
sources['classical']['moszkowski']   = ['http://www.classicalmidi.co.uk/moszk.htm']
sources['classical']['mozart']       = ['http://www.midiworld.com/mozart.htm','http://www.classicalmidi.co.uk/mozart.htm']
sources['classical']['nikolaievich'] = ['http://www.classicalmidi.co.uk/scab.htm']
sources['classical']['orff']         = ['http://www.classicalmidi.co.uk/orff.htm']
sources['classical']['pachelbel']    = ['http://www.classicalmidi.co.uk/pach.htm']
sources['classical']['paderewski']   = ['http://www.classicalmidi.co.uk/paderewski.htm']
sources['classical']['pagg']         = ['http://www.classicalmidi.co.uk/pagg.htm']
sources['classical']['palestrina']   = ['http://www.classicalmidi.co.uk/palestrina.htm']
sources['classical']['paradisi']     = ['http://www.classicalmidi.co.uk/paradisi.htm']
sources['classical']['poulenc']      = ['http://www.classicalmidi.co.uk/poulenc.htm']
sources['classical']['pres']         = ['http://www.classicalmidi.co.uk/pres.htm']
sources['classical']['prokif']       = ['http://www.classicalmidi.co.uk/prokif.htm']
sources['classical']['puccini']      = ['http://www.classicalmidi.co.uk/puccini.htm']
sources['classical']['rachmaninov']  = ['http://www.midiworld.com/rachmaninov.htm','http://www.classicalmidi.co.uk/rach.htm']
sources['classical']['ravel']        = ['http://www.classicalmidi.co.uk/ravel1.htm']
sources['classical']['respig']       = ['http://www.classicalmidi.co.uk/respig.htm']
sources['classical']['rimsky']       = ['http://www.classicalmidi.co.uk/rimsky.htm']
sources['classical']['rossini']      = ['http://www.classicalmidi.co.uk/rossini.htm']
sources['classical']['strauss']     = ['http://www.classicalmidi.co.uk/rstrauss.htm']
sources['classical']['sacrlatt']     = ['http://www.classicalmidi.co.uk/sacrlatt.htm']
sources['classical']['saens']        = ['http://www.classicalmidi.co.uk/saens.htm']
sources['classical']['sanz']         = ['http://www.classicalmidi.co.uk/sanz.htm']
sources['classical']['satie']        = ['http://www.classicalmidi.co.uk/satie.htm']
sources['classical']['scarlatti']    = ['http://www.midiworld.com/scarlatti.htm','http://www.classicalmidi.co.uk/scarlatt.htm']
sources['classical']['schoberg']     = ['http://www.classicalmidi.co.uk/schoberg.htm']
sources['classical']['schubert']     = ['http://www.classicalmidi.co.uk/schubert.htm']
sources['classical']['schumann']     = ['http://www.midiworld.com/schumann.htm', 'http://www.classicalmidi.co.uk/schuman.htm']
sources['classical']['scriabin']     = ['http://www.midiworld.com/scriabin.htm']
sources['classical']['shostakovich'] = ['http://www.classicalmidi.co.uk/shost.htm']
sources['classical']['sibelius']     = ['http://www.classicalmidi.co.uk/sibelius.htm']
sources['classical']['soler']        = ['http://www.classicalmidi.co.uk/soler.htm']
sources['classical']['sor']          = ['http://www.classicalmidi.co.uk/sor.htm']
sources['classical']['sousa']        = ['http://www.classicalmidi.co.uk/sousa.htm']
sources['classical']['stravinsky']   = ['http://www.classicalmidi.co.uk/strav.htm']
sources['classical']['sullivan']     = ['http://www.classicalmidi.co.uk/sull.htm']
sources['classical']['susato']       = ['http://www.classicalmidi.co.uk/susato.htm']
sources['classical']['taktak']       = ['http://www.classicalmidi.co.uk/taktak.htm']
sources['classical']['taylor']       = ['http://www.classicalmidi.co.uk/taylor.htm']
sources['classical']['tchaikovsky']  = ['http://www.midiworld.com/tchaikovsky.htm','http://www.classicalmidi.co.uk/tch.htm']
sources['classical']['thomas']       = ['http://www.classicalmidi.co.uk/thomas.htm']
sources['classical']['vaughan']      = ['http://www.classicalmidi.co.uk/vaughan.htm']
sources['classical']['verdi']        = ['http://www.classicalmidi.co.uk/verdi.htm']
sources['classical']['vivaldi']      = ['http://www.classicalmidi.co.uk/vivaldi.htm']
sources['classical']['wagner']       = ['http://www.classicalmidi.co.uk/wagner.htm']
sources['classical']['walton']       = ['http://www.classicalmidi.co.uk/walton.htm']
sources['classical']['wolf']         = ['http://www.classicalmidi.co.uk/wolf.htm']
sources['classical']['wyschnegradsky'] = ['http://www.classicalmidi.co.uk/Wyschnegradsky.htm']
sources['classical']['yradier']      = ['http://www.classicalmidi.co.uk/yradier.htm']
#sources['classical']['earlymusic']   = ['http://www.midiworld.com/earlymus.htm']








ignore_patterns                      = ['xoom']

file_list = {}
file_list['validation'] = ['classical/handel/hwv333-3.mid', \
                           'classical/mozart/mozk466b.mid', \
                           'classical/byrd/byrd82.mid']
file_list['test']       = ['classical/handel/clangour.mid', \
                           'classical/satie/1749satogv4.mid', \
                           'classical/mozart/mozeine.mid', \
                           'classical/byrd/byrd51.mid', \
                           'classical/scriabin/pr37n3.mid', \
                           'classical/beethoven/515furelse1.mid']

class MusicDataLoader(object):

  def __init__(self, datadir, select_validation_percentage, select_test_percentage):
    self.datadir = datadir
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    self.download_midi_data()
    self.read_data(select_validation_percentage, select_test_percentage)

  def download_midi_data(self):
    """
    download_midi_data will download a number of midi files, linked from the html
    pages specified in the sources dict, into datadir. There will be one subdir
    per genre, and within each genre-subdir, there will be a subdir per composer.
    Hence, similar to the structure of the sources dict.
    """
    midi_files = {}

    if os.path.exists(os.path.join(self.datadir, 'do-not-redownload.txt')):
      print 'Already completely downloaded, delete do-not-redownload.txt to check for files to download.'
      return
    for genre in sources:
      midi_files[genre] = {}
      for composer in sources[genre]:
        midi_files[genre][composer] = []
        for url in sources[genre][composer]:
          print url
          response = urllib2.urlopen(url)
          #if 'classicalmidi' in url:
          #  headers = response.info()
          #  print headers
          data = response.read()

          #htmlinks = re.findall('"(  ?[^"]+\.htm)"', data)
          #for link in htmlinks:
          #  print 'http://www.classicalmidi.co.uk/'+strip(link)
          
          # make urls absolute:
          urlparsed = urlparse.urlparse(url)
          data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
          data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
          #if 'classicalmidi' in url:
          #  print data
          
          links = re.findall('"(http://[^"]+\.mid)"', data)
          for link in links:
            cont = False
            for p in ignore_patterns:
              if p in link:
                print 'Not downloading links with {}'.format(p)
                cont = True
                continue
            if cont: continue
            print link
            filename = link.split('/')[-1]
            valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
            filename = ''.join(c for c in filename if c in valid_chars)
            print genre+'/'+composer+'/'+filename
            midi_files[genre][composer].append(filename)
            localdir = os.path.join(os.path.join(self.datadir, genre), composer)
            localpath = os.path.join(localdir, filename)
            if os.path.exists(localpath):
              print 'File exists. Not redownloading: {}'.format(localpath)
            else:
              try:
                response_midi = urllib2.urlopen(link)
                try: os.makedirs(localdir)
                except: pass
                data_midi = response_midi.read()
                if 'DOCTYPE html PUBLIC' in data_midi:
                  print 'Seems to have been served an html page instead of a midi file. Continuing with next file.'
                elif 'RIFF' in data_midi[0:9]:
                  print 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.'
                else:
                  with open(localpath, 'w') as f:
                    f.write(data_midi)
              except:
                print 'Failed to fetch {}'.format(link)
    with open(os.path.join(self.datadir, 'do-not-redownload.txt'), 'w') as f:
      f.write('This directory is considered completely downloaded.')

  def read_data(self, select_validation_percentage, select_test_percentage):
    """
    read_data takes a datadir with genre subdirs, and composer subsubdirs
    containing midi files, reads them into training data for an rnn-gan model.
    Midi music information will be real-valued frequencies of the
    tones, and intensity taken from the velocity information in
    the midi files.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """
    output_ticks_per_quarter_note = 8.0

    self.genres = sorted(sources.keys())
    print('num genres:{}'.format(len(self.genres)))
    composers = []
    for genre in self.genres:
      composers.extend(sources[genre].keys())
    composers_nodupes = []
    for composer in composers:
      if composer not in composers_nodupes:
        composers_nodupes.append(composer)
    self.composers = composers_nodupes
    self.composers.sort()
    print('num composers:{}'.format(len(self.composers)))

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []

    #max_composers = 2
    #composer_id   = 0
    if select_validation_percentage or select_test_percentage:
      for genre in sources:
        for composer in sources[genre]:
          current_path = os.path.join(self.datadir,os.path.join(genre, composer))
          if not os.path.exists(current_path):
            print 'Path does not exist: {}'.format(current_path)
            continue
          files = os.listdir(current_path)
          filelist = []
          for i,f in enumerate(files):
            if os.path.isfile(os.path.join(current_path,f)):
              filelist.append(os.path.join(os.path.join(genre, composer), f))
      random.shuffle(filelist)
      validation_len = 0
      if select_test_percentage:
        validation_len = int(float(select_validation_percentage/100.0)*len(filelist))
        self.file_list['validation'] = filelist[0:validation_len]
        print ('Selected validation set (FLAG --select_validation_percentage): {}'.format(self.file_list['validation']))
      if select_test_percentage:
        test_len = int(float(select_test_percentage/100.0)*len(filelist))
        self.file_list['test'] = filelist[validation_len:validation_len+test_len]
        print ('Selected test set (FLAG --select_test_percentage): {}'.format(self.file_list['test']))
    
    for genre in sources:
      for composer in sources[genre]:
        current_path = os.path.join(self.datadir,os.path.join(genre, composer))
        if not os.path.exists(current_path):
          print 'Path does not exist: {}'.format(current_path)
          continue
        files = os.listdir(current_path)
        #composer_id += 1
        #if composer_id > max_composers:
        #  print('Only using {} composers.'.format(max_composers))
        #  break
        for i,f in enumerate(files):
          if i % 100 == 99 or i+1 == len(files):
            print 'Reading files {}/{}: {}'.format(genre, composer, i)
          if os.path.isfile(os.path.join(current_path,f)):
            try:
              midi_pattern = midi.read_midifile(os.path.join(current_path,f))
            except:
              print 'Error reading {}'.format(os.path.join(current_path,f))
              continue
            #
            # Interpreting the midi pattern.
            # A pattern has a list of tracks
            # (midi.Track()).
            # Each track is a list of events:
            #   * midi.events.SetTempoEvent: tick, data([int, int, int])
            #     (The three ints are really three bytes representing one integer.)
            #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
            #     (ignored)
            #   * midi.events.KeySignatureEvent: tick, data([int, int])
            #     (ignored)
            #   * midi.events.MarkerEvent: tick, text, data
            #   * midi.events.PortEvent: tick(int), data
            #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
            #   * midi.events.ProgramChangeEvent: tick, channel, data
            #   * midi.events.ControlChangeEvent: tick, channel, data
            #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
            #
            #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
            #     - data[0] is the note (0-127)
            #     - data[1] is the velocity.
            #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
            #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
            #
            #   * midi.events.EndOfTrackEvent: tick(int), data()
            #
            # Ticks are relative.
            #
            # Tempo are in microseconds/quarter note.
            #
            # This interpretation was done after reading
            # http://electronicmusic.wikia.com/wiki/Velocity
            # http://faydoc.tripod.com/formats/mid.htm
            # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
            # and looking at some files. It will hopefully be enough
            # for the use in this project.
            #
            # We'll save the data intermediately with a dict representing each tone.
            # The dicts we put into a list. Times are microseconds.
            # Keys: 'freq', 'velocity', 'begin-tick', 'tick-length'
            #
            # 'Output ticks resolution' are fixed at a 32th note,
            #   - so 8 ticks per quarter note.
            #
            # This approach means that we do not currently support
            #   tempo change events.
            #
            # TODO 1: Figure out pitch.
            # TODO 2: Figure out different channels and instruments.
            #
            
            song_data = []
            
            # Tempo:
            ticks_per_quarter_note = float(midi_pattern.resolution)
            input_ticks_per_output_tick = ticks_per_quarter_note/output_ticks_per_quarter_note
            
            # 'abs_ticks' here are input ticks.
            # Multiply with output_ticks_pr_input_tick for output ticks.
            for track in midi_pattern:
              abs_ticks = 0
              current_note = None
              for event in track:
                abs_ticks += event.tick
                if type(event) == midi.events.SetTempoEvent:
                  pass # These are ignored
                elif (type(event) == midi.events.NoteOffEvent) or \
                     (type(event) == midi.events.NoteOnEvent and \
                      event.velocity == 0):
                  if current_note:
                     #current_note['length'] = float(ticks*microseconds_per_tick)
                     current_note[1] = float(event.tick)/input_ticks_per_output_tick
                     song_data.append(current_note)
                     current_note = None
                elif type(event) == midi.events.NoteOnEvent:
                  begin_tick = float(event.tick)
                  #current_note = {'begin-tick': begin_time, 'tick-length': 0.0, 'freq': tone_to_freq(event.data[0]), 'velocity': float(event.data[1]), 'channel': event.channel}
                  velocity = float(event.data[1])
                  current_note = [begin_tick/input_ticks_per_output_tick, 0.0, tone_to_freq(event.data[0]), velocity, event.channel]
            song_data.sort(key=lambda e: e[0])
            if os.path.join(os.path.join(genre, composer), f) in file_list['validation']:
              self.songs['validation'].append([genre, composer, song_data])
            elif os.path.join(os.path.join(genre, composer), f) in file_list['test']:
              self.songs['test'].append([genre, composer, song_data])
            else:
              self.songs['train'].append([genre, composer, song_data])
          #break
    random.shuffle(self.songs['train'])
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    # TODO REMOVE FOLLOWING LINE. DEBUG: TRYING TO OVERFIT.
    self.songs['train'] = self.songs['train'][0:20]
    return self.songs

  def rewind(self, part='train'):
    self.pointer[part] = 0

  def get_batch(self, batchsize, songlength, part='train'):
    """
      get_batch() returns a batch from self.songs, as a
      list of tensors of dimensions [batchsize, numfeatures]
      the list is of length songlength.
      To have the sequence be the primary index is convention in
      tensorflow's rnn api.
      Songs are currently chopped off after songlength.

      Since self.songs was shuffled in read_data(), the batch is
      a random selection without repetition.

      songlength is related to internal sample frequency.
      We fix this to be every 32th notes. # 50 milliseconds.
      This means 8 samples per quarter note.
      There is currently no notion of tempo in the representation.

      composer and genre is concatenated to each event
      in the sequence. There might be more clever ways
      of doing this. It's not reasonable to change composer
      or genre in the middle of a song.
    """
    if self.pointer[part] > len(self.songs[part])-batchsize:
      return None
    if self.songs[part]:
      batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      # subtract two for start-time and channel, which we don't include.
      numfeatures = len(self.genres)+len(self.composers)+len(batch[0][2][0])-2
      batch_ndarray = np.ndarray(shape=[batchsize, songlength, numfeatures])
      #print 'batch shape: {}'.format(batch_ndarray.shape)
      for s in range(len(batch)):
        songmatrix = np.ndarray(shape=[songlength, numfeatures])
        zeroframe = np.zeros(shape=[len(batch[0][2][0])-2])
        composeronehot = onehot(self.composers.index(batch[s][1]), len(self.composers))
        genreonehot = onehot(self.genres.index(batch[s][0]), len(self.genres))
        eventid = 0
        for n in range(songlength):
          # TODO: channels!
          event = zeroframe
          for e in range(eventid, len(batch[s][2])):
            #if batch[s][2][e]['begin-time'] < n*50000:
            # If people are using the higher resolution of the
            # original midi file, we will get the rounding errors
            # right here. But it will be rounded to 'real' 32th notes.
            if int(batch[s][2][e][0]) == n:
              # we don't include start-time at index 0:
              # and not channel at -1.
              event = np.array(batch[s][2][e][1:-1])
              eventid = e
            elif int(batch[s][2][e][0]) > n:
              # song data lists should have been sorted above.
              break
          songmatrix[n,:] = np.concatenate([genreonehot, composeronehot, event])
        batch_ndarray[s,:,:] = songmatrix
      #batched_sequence = np.split(batch_ndarray, indices_or_sections=songlength, axis=1)
      #return [np.squeeze(s, axis=1) for s in batched_sequence]
      return batch_ndarray
    else:
      raise 'get_batch() called but self.songs is not initialized.'
  
  def get_numfeatures(self):
    return len(self.genres)+len(self.composers)+len(self.songs['train'][0][2][0])-2

def tone_to_freq(tone):
  """
    returns the frequency of a tone. 

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0

def freq_to_tone(freq):
  """
    returns a dict d where
    d['tone'] is the base tone in midi standard
    d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
  int_tone = int(float_tone)
  cents = 1200*math.log(float(freq)/tone_to_freq(int_tone), 2)
  return {'tone': int_tone, 'cents': cents}

def onehot(i, length):
  a = np.zeros(shape=[length])
  a[i] = 1
  return a

