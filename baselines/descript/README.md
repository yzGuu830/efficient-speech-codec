## Descript's Audio Codec (DAC) Experimental Reproduction

This folder is mostly borrowed from [Descript's Github Repository](https://github.com/descriptinc/descript-audio-codec).

We adapt a few features for customized reproduction. For developmental setups, refer to the original repository.


## Reproduce DAC Baselines

```ruby
torchrun --nproc_per_node gpu train_customize.py --config 16kHz_dns_9k.yml
```
This reproduces 16kHz (0.5kbps ~ 9.0kbps) DAC with adversarial setups.

```ruby
torchrun --nproc_per_node gpu train_customize_no_adv.py --config 16kHz_dns_9k_tiny.yml
```
This reproduces 16kHz (0.5kbps ~ 9.0kbps) DAC in non-adversarial setups.