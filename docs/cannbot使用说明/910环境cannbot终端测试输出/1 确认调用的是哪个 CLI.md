developer@5989a23a0e2c433b87152c2132a9c46d:/mnt/workspace$ cd /mnt/workspace/pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622
loper/.nvm/versions/node/v18.19.1/bin/cannbot

"$CANNBOT_BIN" --version
"$CANNBOT_BIN" --helpdeveloper@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ 
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ CANNBOT_BIN=/home/developer/.nvm/versions/node/v18.19.1/bin/cannbot
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ 
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ "$CANNBOT_BIN" --version

1.1.2
developer@5989a23a0e2c433b87152c2132a9c46d:.../pypto-pro-workflow/work/fused_conv_sigmoid_20260928_221622$ "$CANNBOT_BIN" --help
▄██████▄   ████   ██▄   ██ ██▄   ██ ███████▄           ▄▄  
██    ██  ██  ██  ████  ██ ████  ██ ██    ██ ▄█████▄ ▄▄██▄▄
██       ██    ██ ██ ██ ██ ██ ██ ██ ███████  ██   ██ ▀▀██▀▀
██    ██ ████████ ██  ████ ██  ████ ██    ██ ██   ██   ██  
▀██████▀ ██    ██ ██   ▀██ ██   ▀██ ███████▀ ▀█████▀   ▀███

Commands:
  cannbot completion          generate shell completion script
  cannbot acp                 start ACP (Agent Client Protocol) server
  cannbot mcp                 manage MCP (Model Context Protocol) servers
  cannbot [project]           start CANNBot tui                                            [default]
  cannbot attach <url>        attach to a running CANNBot server
  cannbot run [message..]     run CANNBot with a message
  cannbot debug               debugging and troubleshooting tools
  cannbot providers           manage AI providers and credentials                    [aliases: auth]
  cannbot agent               manage agents
  cannbot upgrade [target]    upgrade CANNBot to the latest or a specific version
  cannbot uninstall           uninstall CANNBot and remove all related files
  cannbot serve               starts a headless CANNBot server
  cannbot web                 start CANNBot server and open web interface
  cannbot models [provider]   list all available models
  cannbot stats               show token usage and cost statistics
  cannbot export [sessionID]  export session data as JSON
  cannbot import <file>       import session data from JSON file or URL
  cannbot github              manage GitHub agent
  cannbot pr <number>         fetch and checkout a GitHub PR branch, then run CANNBot
  cannbot session             manage sessions
  cannbot plugin <module>     install plugin and update config                       [aliases: plug]
  cannbot connect             connect to CANNBot AI gateway
  cannbot db                  database tools

Positionals:
  project  path to start CANNBot in                                                         [string]

Options:
  -h, --help          show help                                                            [boolean]
  -v, --version       show version number                                                  [boolean]
      --print-logs    print logs to stderr                                                 [boolean]
      --log-level     log level                 [string] [choices: "DEBUG", "INFO", "WARN", "ERROR"]
      --pure          run without external plugins                                         [boolean]
      --port          port to listen on                                        [number] [default: 0]
      --hostname      hostname to listen on                          [string] [default: "127.0.0.1"]
      --mdns          enable mDNS service discovery (defaults hostname to 0.0.0.0)
                                                                          [boolean] [default: false]
      --mdns-domain   custom domain name for mDNS service (default: opencode.local)
                                                                [string] [default: "opencode.local"]
      --cors          additional domains to allow for CORS                     [array] [default: []]
  -m, --model         model to use in the format of provider/model                          [string]
  -c, --continue      continue the last session                                            [boolean]
  -s, --session       session id to continue                                                [string]
      --fork          fork the session when continuing (use with --continue or --session)  [boolean]
      --prompt        prompt to use                                                         [string]
      --agent         agent to use                                                          [string]
      --mini          start the minimal interactive interface             [boolean] [default: false]
      --no-replay     disable mini session history replay on resume and after resize       [boolean]
      --replay-limit  cap visible mini replay to the newest N messages                      [number]