

# if __name__ == '__main__':

#     encoder = CSVQ_Encoder()
#     decoder = CSVQ_Decoder()

#     T, F = 601, 201
#     X = torch.randn(4,2,F,T)
#     print('raw_feature: ', X.shape)

#     print('encoded_features: ')
#     encoded_hs = encoder(X)
#     for i in range(len(encoded_hs)):
#         print(i, encoded_hs[i].shape)

#     print('decoded_features: ')
#     recon, vq_loss = decoder(encoded_hs)
#     print(recon.shape, vq_loss)

if __name__ == '__main__':
    import torchaudio
    print(torchaudio.__version__)
