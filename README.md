# Trend__seeker bot
WIP - create a Twitter bot to engage and interact with humans

* temporarily on hold until the politics die down on Twitter. In the meantime I'm saving all the wordclouds and data to do some machine learning on later.


To run in the background on OSX:

# twitterBot.plist
in ~/Library/LaunchDaemons create a name.plist file
    
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
      <key>Label</key>
      <string>com.timestocome.twittercloud</string>

      <key>ProgramArguments</key>
      <array>
        <string>python</string>
        <string>/Users/yourname/directoryOfTwitterBotProgram/TwitterCloud/CollectAndProcessTweets.py</string>
      </array>

      <key>Nice</key>
      <integer>1</integer>

      <key>StartInterval</key>
      <integer>14400</integer>
  
      <key>RunAtLoad</key>
      <true/>

      <key>StandardErrorPath</key>
      <string>/Users/yourname/directoryOfTwitterBotProgram/TwitterCloud/TwitterBot.err</string>

      <key>StandardOutPath</key>
      <string>/Users/yourname/directoryOfTwitterBotProgram/TwitterCloud/TwitterBot.out</string>
    </dict>
    </plist>


# to start the twitter bot
launchctl load /Users/yourname/Library/LaunchDaemons/twitterBot.plist

