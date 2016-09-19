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
sources['classical']['albinoni']      = ['http://www.classicalmidi.co.uk/albinoni.htm']
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
sources['classical']['haydn']        = ['http://www.midiworld.com/haydn.htm','http://www.classicalmidi.co.uk/haydn.htm']
sources['classical']['handel']       = ['http://www.midiworld.com/handel.htm','http://www.classicalmidi.co.uk/handel.htm']
sources['classical']['hummel']       = ['http://www.midiworld.com/hummel.htm','http://www.classicalmidi.co.uk/hummel.htm']
sources['classical']['liszt']        = ['http://www.midiworld.com/liszt.htm','http://www.classicalmidi.co.uk/liszt.htm']
sources['classical']['mendelssohn']  = ['http://www.midiworld.com/mendelssohn.htm','http://www.classicalmidi.co.uk/mend.htm']
sources['classical']['mozart']       = ['http://www.midiworld.com/mozart.htm','http://www.classicalmidi.co.uk/mozart.htm']
sources['classical']['rachmaninov']  = ['http://www.midiworld.com/rachmaninov.htm','http://www.classicalmidi.co.uk/rach.htm']
sources['classical']['ravel']        = ['http://www.classicalmidi.co.uk/ravel1.htm']
sources['classical']['satie']        = ['http://www.classicalmidi.co.uk/satie.htm']
sources['classical']['scarlatti']    = ['http://www.midiworld.com/scarlatti.htm','http://www.classicalmidi.co.uk/scarlatt.htm']
sources['classical']['schubert']     = ['http://www.classicalmidi.co.uk/schubert.htm']
sources['classical']['schumann']     = ['http://www.midiworld.com/schumann.htm']
sources['classical']['scriabin']     = ['http://www.midiworld.com/scriabin.htm']
sources['classical']['shostakovich'] = ['http://www.classicalmidi.co.uk/shost.htm']
sources['classical']['tchaikovsky']  = ['http://www.midiworld.com/tchaikovsky.htm','http://www.classicalmidi.co.uk/tch.htm']
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

  def __init__(self, datadir):
    self.datadir = datadir
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    self.download_midi_data()
    self.read_data()

  def download_midi_data(self):
    """
    download_midi_data will download a number of midi files, linked from the html
    pages specified in the sources dict, into datadir. There will be one subdir
    per genre, and within each genre-subdir, there will be a subdir per composer.
    Hence, similar to the structure of the sources dict.
    """
    midi_files = {}

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

          # make urls absolute:
          urlparsed = urlparse.urlparse(url)
          data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
          data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
          #if 'classicalmidi' in url:
          #  print data
          
          links = re.findall('"(http://[^"]+\.mid)"', data)
          htmlinks = re.findall('"(http://[^"]+\.htm)"', data)
          for link in htmlinks:
            print 'html '+link
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
                elif data_midi[0:4] == "RIFF":
                  print 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.'
                else:
                  with open(localpath, 'w') as f:
                    f.write(data_midi)
              except:
                print 'Failed to fetch {}'.format(link)

  def read_data(self):
    """
    read_data takes a datadir with genre subdirs, and composer subsubdirs
    containing midi files, reads them into training data for an rnn-gan model.
    Midi music information will be real-valued frequencies of the
    tones, and intensity taken from the velocity information in
    the midi files.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.
    """
    self.genres = sorted(sources.keys())
    num_genres = len(self.genres)
    composers = []
    for genre in self.genres:
      composers.append(sources[genre].keys())
    composers_nodupes = []
    for composer in composers:
      if composer not in composers_nodupes:
        composers_nodupes.append(composer)
    self.composers = composers_nodupes
    self.composers.sort()
    self.num_composers = len(self.composers)

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []

    for genre in sources:
      for composer in sources[genre]:
        current_path = os.path.join(self.datadir,os.path.join(genre, composer))
        if not os.path.exists(current_path):
          print 'Path does not exist: {}'.format(current_path)
          continue
        files = os.listdir(current_path)
        for i,f in enumerate(files):
          if i % 100 == 99 or i+1 == len(files):
            print 'Reading files {}/{}: {}'.format(genre, composer, i)
          if os.path.isfile(os.path.join(current_path,f)):
            try:
              midi_pattern = midi.read_midifile(os.path.join(current_path,f))
            except:
              print 'Error reading {}'.format(os.path.join(current_path,f))
              continue
            song_data = []
            tempo_events = []
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
            # Keys: 'freq', 'velocity', 'begin-time', 'length'
            #
            # TODO: Figure out pitch.
            #
            # Tempo:
            ticks_per_quarter_note = midi_pattern.resolution
            for track in midi_pattern:
              ticks = 0
              for event in track:
                ticks += event.tick
                if type(event) == midi.events.SetTempoEvent:
                  tempo_events.append({'abs_tick': ticks, 'microseconds_per_quarter_note': event.data[2]+event.data[1]*256+event.data[0]*256*256})
            tempo_events.sort(key=lambda e: e['abs_tick'])
            if tempo_events:
              microseconds_per_quarter_note = tempo_events[0]['microseconds_per_quarter_note']
            else:
              microseconds_per_quarter_note = 500000
            microseconds_per_tick = float(microseconds_per_quarter_note) / float(ticks_per_quarter_note)

            for track in midi_pattern:
              start_time_of_current_ticking = 0
              ticks = 0
              abs_ticks = 0
              next_tempo_event = 1
              current_note = None
              for event in track:
                if len(tempo_events) > next_tempo_event \
                   and tempo_events[next_tempo_event]['abs_tick'] < abs_ticks+event.tick:
                  ticks_in_new_tempo = abs_ticks+event.tick-tempo_events[next_tempo_event]['abs_tick']
                  start_time_of_current_ticking += microseconds_per_tick*(tempo_events[next_tempo_event]['abs_tick']-abs_ticks)

                  ticks = ticks_in_new_tempo
                  microseconds_per_tick = float(tempo_events[next_tempo_event]['microseconds_per_quarter_note']) / float(ticks_per_quarter_note)
                  next_tempo_event += 1
                else:
                  ticks += event.tick
                abs_ticks += event.tick
                if type(event) == midi.events.SetTempoEvent:
                  pass # These were already handled in a previous loop
                elif (type(event) == midi.events.NoteOffEvent) or \
                     (type(event) == midi.events.NoteOnEvent and \
                      event.velocity == 0):
                  if current_note:
                     #current_note['length'] = float(ticks*microseconds_per_tick)
                     current_note[1] = float(ticks*microseconds_per_tick)
                     song_data.append(current_note)
                     current_note = None
                elif type(event) == midi.events.NoteOnEvent:
                  begin_time = float(ticks*microseconds_per_tick+start_time_of_current_ticking)
                  #current_note = {'begin-time': begin_time, 'length': 0.0, 'freq': tone_to_freq(event.data[0]), 'velocity': float(event.data[1]), 'channel': event.channel}
                  velocity = float(event.data[1])
                  current_note = [begin_time, 0.0, tone_to_freq(event.data[0]), velocity, event.channel]
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
      We fix this to be every 50 milliseconds.
      This means 20 samples per second.
      This might be subject to change, if ticks
      or parts of notes should be basis for the notion of time.

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
      numfeatures = len(self.genres)+len(self.composers)+len(batch[0][2][0])-2
      # subtract two for start-time and channel, which we don't include.
      batch_ndarray = np.ndarray(shape=[batchsize, songlength, numfeatures])
      print 'batch shape: {}'.format(batch_ndarray.shape)
      for b in range(len(batch)):
        for s in range(len(batch[b])):
          songmatrix = np.ndarray(shape=[songlength, numfeatures])
          zeroframe = np.zeros(shape=[len(batch[0][2][0])-2])
          composeronehot = onehot(self.composers.index(batch[b][s][1]), len(self.composers))
          genreonehot = onehot(self.genres.index(batch[b][s][0]), len(self.genres))
          evendid = 0
          for n in range(songlength):
            # TODO: channels!
            event = zeroframe
            for e in range(eventid, len(batch[b][s][2])):
              if batch[b][s][2][e]['begin-time'] < n*50000:
                # we don't include start-time at index 0:
                event = np.array(batch[b][s][2][e][1:-1])
                eventid = e
            songmatrix[n,:] = np.concatenate([genreonehot, composeronehot, event])
          batch_ndarray[b,:,:] = songmatrix
      batched_sequence = np.split(batch_ndarray, indices_or_sections=songlength, axis=1)
      return [np.squeeze(s, axis=1) for s in batched_sequence]
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

