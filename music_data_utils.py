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


import urlparse, urllib2, os, midi, math, random
import numpy as np

sources                              = {}
sources['classical']                 = {}
sources['classical']['misc']         = ['http://www.midiworld.com/classic.htm']
sources['classical']['alkan']        = ['http://www.classicalmidi.co.uk/alkan.htm'] #'http://www.kunstderfuge.com/alkan.htm']
sources['classical']['bach']         = ['http://www.midiworld.com/bach.htm','http://www.classicalmidi.co.uk/bach.htm']
                                       # ,'http://www.kunstderfuge.com/bach/harpsi.htm','http://www.kunstderfuge.com/bach/wtk2.htm','http://www.kunstderfuge.com/bach/wtk1.htm','http://www.kunstderfuge.com/bach/organ.htm','http://www.kunstderfuge.com/bach/chamber.htm','http://www.kunstderfuge.com/bach/canons.htm','http://www.kunstderfuge.com/bach/chorales.htm','http://www.kunstderfuge.com/bach/variae.htm']
sources['classical']['bartok']       = ['http://www.midiworld.com/bartok.htm','http://www.classicalmidi.co.uk/bartok.htm']
sources['classical']['beethoven']    = ['http://www.midiworld.com/beethoven.htm','http://www.classicalmidi.co.uk/beethoven.htm']
                                       #,'http://www.kunstderfuge.com/beethoven/klavier.htm','http://www.kunstderfuge.com/beethoven/chamber.htm','http://www.kunstderfuge.com/beethoven/variae.htm']
sources['classical']['brahms']       = ['http://www.midiworld.com/brahms.htm','http://www.classicalmidi.co.uk/brahms.htm']
                                       #,'http://www.kunstderfuge.com/brahms.htm']
sources['classical']['byrd']         = ['http://www.midiworld.com/byrd.htm','http://www.classicalmidi.co.uk/byrd.htm']
                                       #,'http://www.kunstderfuge.com/byrd.htm']
sources['classical']['chopin']       = ['http://www.midiworld.com/chopin.htm','http://www.classicalmidi.co.uk/chopin.htm']
                                       #,'http://www.kunstderfuge.com/chopin.htm']
sources['classical']['debussy']      = ['http://www.classicalmidi.co.uk/debussy.htm'] #'http://www.kunstderfuge.com/debussy.htm']
sources['classical']['haydn']        = ['http://www.midiworld.com/haydn.htm','http://www.classicalmidi.co.uk/haydn.htm']
                                       #,'http://www.kunstderfuge.com/haydn.htm']
sources['classical']['handel']       = ['http://www.midiworld.com/handel.htm','http://www.classicalmidi.co.uk/handel.htm']
                                       #,'http://www.kunstderfuge.com/handel.htm']
sources['classical']['hummel']       = ['http://www.midiworld.com/hummel.htm','http://www.classicalmidi.co.uk/hummel.htm']
sources['classical']['liszt']        = ['http://www.midiworld.com/liszt.htm','http://www.classicalmidi.co.uk/liszt.htm']
                                       #,'http://www.kunstderfuge.com/liszt.htm']
sources['classical']['mendelssohn']  = ['http://www.midiworld.com/mendelssohn.htm','http://www.classicalmidi.co.uk/mend.htm']
                                       #,'http://www.kunstderfuge.com/mendelssohn.htm']
sources['classical']['mozart']       = ['http://www.midiworld.com/mozart.htm','http://www.classicalmidi.co.uk/mozart.htm']
                                       #,'http://www.kunstderfuge.com/mozart.htm']
sources['classical']['rachmaninov']  = ['http://www.midiworld.com/rachmaninov.htm','http://www.classicalmidi.co.uk/rach.htm']
#sources['classical']['raff']         = ['http://www.kunstderfuge.com/raff.htm']
sources['classical']['ravel']        = ['http://www.classicalmidi.co.uk/ravel1.htm'] #'http://www.kunstderfuge.com/ravel.htm']
sources['classical']['satie']        = ['http://www.classicalmidi.co.uk/satie.htm'] #'http://www.kunstderfuge.com/satie.htm']
sources['classical']['scarlatti']    = ['http://www.midiworld.com/scarlatti.htm','http://www.classicalmidi.co.uk/scarlatt.htm']
sources['classical']['schubert']     = ['http://www.classicalmidi.co.uk/schubert.htm'] #'http://www.kunstderfuge.com/schubert.htm']
sources['classical']['schumann']     = ['http://www.midiworld.com/schumann.htm']
                                       #,'http://www.kunstderfuge.com/schumann.htm']
sources['classical']['scriabin']     = ['http://www.midiworld.com/scriabin.htm']
sources['classical']['shostakovich'] = ['http://www.classicalmidi.co.uk/shost.htm'] #'http://www.midiworld.com/shostakovich.htm']
sources['classical']['tchaikovsky']  = ['http://www.midiworld.com/tchaikovsky.htm','http://www.classicalmidi.co.uk/tch.htm']
                                       #,'http://www.kunstderfuge.com/tchaikovsky.htm']
sources['classical']['earlymusic']   = ['http://www.midiworld.com/earlymus.htm']

ignore_patterns                      = ['xoom']

class MusicDataLoader(object):

  def __init__(self, datadir):
    self.datadir = datadir
    download_midi_data()
    read_data()

  def download_midi_data():
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
          if 'classicalmidi' in url:
            headers = response.info()
            print headers
          data = response.read()

          # make urls absolute:
          urlparsed = urlparse.urlparse(url)
          data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
          data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
          if 'classicalmidi' in url:
            print data
          
          links = re.findall('"(http://[^"]+\.mid)"', data)
          #print links
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
                if 'DOCTYPE html PUBLIC' not in data_midi:
                  with open(localpath, 'w') as f:
                    f.write(data_midi)
                else:
                  print 'Seems to have been served an html page instead of a midi file. Continuing with next file.'
              except:
                print 'Failed to fetch {}'.format(link)

  def read_data():
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
    for genre in genres:
      composers.append(sources[genre].keys())
    composers_nodupes = []
    for composer in composers:
      if composer not in composers_nodupes:
        composers_nodupes.append(composer)
    self.composers = composers_nodupes
    self.composers.sort()
    self.num_composers = len(self.composers)

    self.songs = []

    for genre in sources:
      for composer in sources[genre]:
        current_path = os.path.join(self.datadir,os.path.join(genre, composer))
        files = os.listdir(current_path)
        for i,f in enumerate(files):
          if i % 100 == 0: print 'Reading files: {}'.format(i)
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
                  current_note = [begin_time, 0.0, tone_to_freq(event.data[0]), velocity, event.channel}
            self.songs.append([genre, composer, song_data])
          #break
    return self.songs

  def get_batch(batchsize, songlength):
    """
      get_batch() returns a random selection from self.songs, as a
      list of tensors of dimensions [batchsize, numfeatures]
      the list is of length songlength.
      songs are chopped off after songlength.

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
    if self.songs:
      batch = random.sample(self.songs, batchsize)
      numfeatures = len(self.genres)+len(self.composers)+len(batch[0][2][0])-2
      # subtract two for start-time and channel, which we don't include.
      batch_ndarray = np.ndarray(shape=[batchsize, songlength, numfeatures)
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
    return len(self.genres)+len(self.composers)+len(self.songs[0][2][0])-2

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
