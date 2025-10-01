# Configuration file for veloxq-api-sdk.

c = get_config()  #noqa

c.VeloxQAPIConfig.token = 'b0c14738-36e0-4364-9066-a2a54dc02d7e'  # --VeloxQAPIConfig.token=<Unicode>
                              #     API token for authentication with the VeloxQ service.
                              #     Default: ''

c.VeloxQAPIConfig.url = 'https://api-dev.veloxq.com'  # --VeloxQAPIConfig.url=<Unicode>
                                                      #     Base URL for the VeloxQ API.
                                                      #     Default: 'https://api-dev.veloxq.com'
