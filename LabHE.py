# LabHE implementation from https://eprint.iacr.org/2017/326
# Uses Paillier as underlying additively homomorphic encryption and Keccak as the PRF

import random
import hashlib
import math
import sys
import numpy
from gmpy2 import mpz
import paillier
import util_fpv

try:
    import gmpy2
    HAVE_GMP = True
except ImportError:
    HAVE_GMP = False

DEFAULT_KEYSIZE = 1024
"""We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative. 
The range N/3 < x < 2N/3 allows for overflow detection."""


"""The LabHE 'public key' is composed from the Paillier public key and the client's
secret key. It is not a public key in the usual sense because a secret key is needed 
in order to encrypt a message - through the PRF. The LabHE private key is composed 
of the Paillier secret key and the clients' public keys"""

def generate_LabHE_keypair(usk, n_length=DEFAULT_KEYSIZE):
    """Return a new :class:`LabHEPublicKey` and :class:`LabHEPrivateKey`.
    Composed from a master public key mpk and a master secret key msk that 
    corespond to the Paillier keys
    """
    mpk, msk = paillier.generate_paillier_keypair(None,n_length)
    # lsk = msk, mpk.encrypt(usk)
    # lpk = mpk, usk

    lpk = LabHEPublicKey(mpk)
    if len(usk) == 1:
        lsk = LabHEPrivateKey(msk, mpk.encrypt(usk))
    else:
        lsk = LabHEPrivateKey(msk, util_fpv.encrypt_vector(mpk,usk))

    return lpk, lsk

class LabHEPublicKey(object):
    """Composed of PaillierPublicKey 

    Attributes:
      mpk (PaillierPublicKey): the public key of the underlying Paillier scheme
      max_int (int): n/3, the maximum positive value a plaintext can be
    """
    def __init__(self, mpk):
        self.Pai_key = mpk
        self.n = mpk.n
        self.max_int = mpk.n // 3 - 1

    def offline_gen_secret(self, label, usk):
        """LabHE  generation of a secret from a label 
            with a pseudorandom function, in this case sha3_224. Change the 
        PRF as desired.

        Args:
          plaintext (label): a positive integer < :attr:`n` that uniquely 
          identifies a plaintext that will be encrypted online.
            unique
          usk (int): a positive integer representing the key for a PRF.

        Returns:
          int: secret = PRF(usk,label)

        """
        self.usk = usk  
        hash = hashlib.sha3_224()
        hash.update(('%s%s' % (bin(usk).encode('utf-8'), bin(label).encode('utf-8'))).encode('utf-8'))
        secret = int(hash.hexdigest(),16)
        return secret

    def offline_encrypt(self, secret):
        """LabHE offline encryption of the secret"""

    def encrypt(self, plaintext, secret, enc_secret=None, r_value=None):
        """LabHE online encryption of a positive integer plaintext < :attr:`n`.

        Args:
          plaintext (int): a positive integer < :attr:`n` to be splitted 
            between the secret below and plaintext - secret
          secret (int): a positive integer < :attr:`n which will be 
            Paillier encrypted
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            r_value is None), a random value is used.

        Returns:
          int: splitted plaintext: plaintext - secret,
          int: Paillier encryption of secret.

        Raises:
          TypeError: if plaintext, secret are not an int or mpz.
        """
        if not isinstance(secret, int) and not isinstance(secret, type(mpz(1))) and not isinstance(secret, numpy.integer):
            raise TypeError('Expected int type secret but got: %s' %
                            type(secret))
        if not isinstance(plaintext, int) and not isinstance(plaintext, type(mpz(1))) and not isinstance(plaintext, numpy.integer):
            raise TypeError('Expected int type plaintext but got: %s' %
                            type(plaintext))
        if not isinstance(enc_secret, paillier.EncryptedNumber) and enc_secret is not None:
            raise TypeError('Expected encrypted secret to be type Paillier.EncryptedNumber or None but got: %s' %
                            type(enc_secret))
        if enc_secret is None:
            ciphertext = plaintext - secret, self.Pai_key.encrypt(secret,r_value)
        else:
            ciphertext = plaintext - secret, enc_secret
        encrypted_number = LabEncryptedNumber(self, ciphertext)
        return encrypted_number


class LabHEPrivateKey(object):
    """Contains the Paillier private key, the client public key and associated decryption method.

    Args:
      public_key (:class:`PaillierPublicKey`): The corresponding public
        key.
      p (int): private secret - see Paillier's paper.
      q (int): private secret - see Paillier's paper.

    Attributes:
      msk (PaillierPrivateKey): The private key of the underlying Paillier scheme.
      upk (int): Client's public key - can be a vector if there are more clients
      usk (int): Client's secret key - can be a vector if there are more clients
      n (int): Paillier modulus
      
    """
    def __init__(self, msk, upk):
        self.msk = msk
        self.upk = upk
        if len(upk) == 1:
            self.usk = msk.decrypt(upk)
        else:
            self.usk = util_fpv.decrypt_vector(msk,upk)
        self.n = msk.n
        self.mpk = msk.public_key

    def __repr__(self):
        pub_repr = repr(self.mpk)
        return "<LabHEPrivateKey for {}>".format(pub_repr)

    def decrypt(self, encrypted_number, secret=None):
        """Return the decrypted & decoded plaintext of *encrypted_number*.

        Args:
          encrypted_number (LabEncryptedNumber): an
            :class:`LabEncryptedNumber` with a public key that matches this
            private key.

        Returns:
          the int or float that `LabEncryptedNumber` was holding. 

        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`LabEncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        if not isinstance(encrypted_number, LabEncryptedNumber) and not isinstance(encrypted_number, paillier.EncryptedNumber): 
            raise TypeError('Expected encrypted_number to be an LabEncryptedNumber or paillier.EncryptedNumber'
                            ' not: %s' % type(encrypted_number))

        
        if isinstance(encrypted_number, LabEncryptedNumber):
            if self.mpk != encrypted_number.mpk:
                raise ValueError('encrypted_number was encrypted against a '
                                 'different key!')

            if secret is None:
                if len(encrypted_number.ciphertext) == 2:
                    secret = self.raw_offline_decrypt(encrypted_number.ciphertext[1])
                    ciphertext = encrypted_number.ciphertext[0]
                else:
                    if len(encrypted_number.ciphertext) == 1:
                        raise TypeError('Expected a secret as an input')
            else:
                if isinstance(secret, paillier.EncryptedNumber):
                    secret = self.raw_offline_decrypt(secret)
                # secret = self.raw_offline_decrypt(encrypted_number.ciphertext[1])+secret
            # Need to distinguish when a secret is given as a plaintext to avoid decryption and
            # when another secret is given to make up for a function applied over the secrets that 
            # correspond with a Paillier ciphertext. Until then, add the secret separately
            
            ciphertext = encrypted_number.ciphertext[0]
        else:
            if secret is None:
                raise TypeError('Expected a secret as an input')
            else:
                if isinstance(secret, paillier.EncryptedNumber):
                    secret = self.raw_offline_decrypt(secret)
            ciphertext = self.msk.decrypt(encrypted_number)

        return self.raw_decrypt(ciphertext, secret)


    def raw_decrypt(self, ciphertext, secret):
        """Decrypt raw ciphertext and return raw plaintext.

        Args:
          ciphertext (int): plaintext - secret, second part 
            is encrypted secret (usually from :meth:`raw_offline_decrypt`)
            that is to be Paillier decrypted.

        Returns:
          int: decryption of the LabHE ciphertext. This is a positive
          integer < :attr:`public_key.n`.

        Raises:
          TypeError: if the ciphertext is not int.
        """
        if not isinstance(ciphertext, int) and not isinstance(ciphertext, type(mpz(1))) and not isinstance(ciphertext, numpy.int64):
            raise TypeError('Expected ciphertext to be an int, not: %s' %
                type(ciphertext))


        value = ciphertext + secret
        if value < self.n/3:
            return int(value)
        else:
            return int(value - self.n)

    def raw_offline_decrypt(self, encr_secret):
        """Offline decryption of the secret.

        Args:
          encr_secret (int): Paillier encryption of result of the 
          program ran on the input secrets, that is to be Paillier decrypted.

        Returns:
          int: secret. This is a positive integer < :attr:`public_key.n`.

        Raises:
          TypeError: if the encr_secret are not int.
        """
        secret = self.msk.decrypt(encr_secret)
        return secret

class LabEncryptedNumber(object):
    """Represents the LabHE encryption of an int.

    Typically, an `LabEncryptedNumber` is created by
    :meth:`LabHEPublicKey.encrypt`. You would only instantiate an
    `LabEncryptedNumber` manually if you are de-serializing a number
    someone else encrypted.


    LabHE encryption and the Paillier underlying scheme are only 
    defined for non-negative integers less than 
    :attr:`PaillierPublicKey.n`. :class:`EncodedNumber` provides
    an encoding scheme for and signed integers that is compatible 
    with the partially homomorphic properties of the Paillier
    cryptosystem:

    1. D(E(a) * E(b)) = a + b
    2. D(E(a)**b)     = a * b

    where `a` and `b` are ints, `E` represents encoding then
    encryption, and `D` represents decryption then decoding for 
    the Paillier scheme.

    The LabHE scheme allows additions and multiplications between 
    encrypted data and/or plaintext data.

    The extended LabHE scheme allows the evaluation of degree-d 
    polynomials on encrypted data, as long as the extra data 
    necessary is provided, see paper Alexandru et al 2018.

    Args:
      mpk (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.
      ciphertext (int, int ): encrypted representation of the encoded 
        number.

    Attributes:
      mpk (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.

    Raises:
      TypeError: if *ciphertext* is not an (int, int), or if *public_key* is
        not a :class:`PaillierPublicKey`.
    """

    ####### There are two types of ciphertexts, beware! LabEncryptedNumber and paillier.EncryptedNumber
    def __init__(self, mpk, ciphertext):
        self.mpk = mpk
        self.ciphertext = ciphertext
        if isinstance(self.ciphertext, LabEncryptedNumber) | isinstance(self.ciphertext, paillier.EncryptedNumber):
            raise TypeError('Ciphertext should be an integer')
        if not isinstance(self.mpk, LabHEPublicKey):
            raise TypeError('mpk should be a LabHEPublicKey')

    def __add__(self, other):
        """Add an int, `LabEncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, LabEncryptedNumber) | isinstance(other, paillier.EncryptedNumber):
            return self._add_encrypted(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <LabEncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply by an int or two full `LabEncryptedNumbers`."""
        if isinstance(other, LabEncryptedNumber):
            product = self._mul_encrypted(other)
        else:
            if other < 0:
                other = other[0] + self.mpk.n
            product = self._mul_scalar(other)
        # return LabEncryptedNumber(self.mpk, product.ciphertext)
        return product

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    # def ciphertext(self):
    #     """Return the ciphertext of the LabEncryptedNumber.

    #     Returns:
    #       int, int , the ciphertext. 
    #     """
    #     return self.ciphertext

    def _add_scalar(self, scalar):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          scalar: an int b, to be added to `self`.

        Returns:
          LabEncryptedNumber: if E(a) = (a-s,[[s]]), return 
          (a + b - s,[[s]]), else if E(a) = [[a]], return 
          Pai_add([[a]],[[b]])
        """

        a, b = self.ciphertext, scalar

        if len(a)==2:
            sum_ciphertext = a[0]+b, a[1]
        else:
            sum_ciphertext = a + b # The override of sum is taken care in Paillier
        return LabEncryptedNumber(self.mpk, sum_ciphertext)

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).

        Args:
          other (LabEncryptedNumber): an `LabEncryptedNumber` to add to self.

        Returns:
          LabEncryptedNumber: if E(a) = (a-s,[[s]]) and E(b) = (b-t,[[t]]), 
          return (a + b - s - t,Pai_add([[s]],[[t]])), else if E(a) = 
          (a - s,[[s]]) and E(b) = [[b]], return (a - s, Pai_add([[s]],[[b]])), 
          else if E(a) = [[a]] and E(b) = (b - s,[[s]]) , return 
          (b - s, Pai_add([[s]],[[a])), else if E(a) = [[a]] and E(b) = [[b]], 
          return Pai_add([[a]],[[b]])

        Raises:
          ValueError: if numbers were encrypted against different keys.
        """

        if isinstance(self, LabEncryptedNumber) & isinstance(other, LabEncryptedNumber):
            if self.mpk != other.mpk :
                raise ValueError("Attempted to add numbers encrypted against "
                                "different public keys!")
            a, b = self.ciphertext, other.ciphertext
            sum_ciphertext = a[0] + b[0], a[1] + b[1]
        else:
            if isinstance(self, LabEncryptedNumber) & isinstance(other, paillier.EncryptedNumber):
                if self.mpk.Pai_key != other.public_key :
                    raise ValueError("Attempted to add numbers encrypted against "
                                    "different public keys!")
                a, b = self.ciphertext, other
                len_a = len(a)
                if len_a == 2:
                    sum_ciphertext = a[0], a[1] + b
                else:
                    sum_ciphertext = a + b
            else:
                    if isinstance(other, LabEncryptedNumber) & isinstance(self, paillier.EncryptedNumber):
                        if other.mpk.Pai_key != self.public_key :
                            raise ValueError("Attempted to add numbers encrypted against "
                                            "different public keys!")
                        a, b = other.ciphertext, self
                        len_b = len(b)
                        if len_b == 2:
                            sum_ciphertext = b[0], a + b[1]
                        else:
                            sum_ciphertext = a + b

        return LabEncryptedNumber(self.mpk, sum_ciphertext)


    def _mul_scalar(self, plaintext):
        """Returns the E(a * plaintext), where E(a) = ciphertext

        Args:
          plaintext (int): number by which to multiply the
            `LabEncryptedNumber`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`

        Returns:
          LabEncryptedNumber: if E(a) = (a-s,[[s]]), return ((a-s)b,Pai_mult([[s]],b)),
            else if E(a) = [[a]], return Pai_mult([[a]],b)

        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if not isinstance(plaintext, int) and not isinstance(plaintext, type(mpz(1))) and not isinstance(plaintext, numpy.int64):
            raise TypeError('Expected ciphertext to be int, not %s' %
                type(plaintext))

        if plaintext < 0 or plaintext >= self.mpk.n:
            raise ValueError('Scalar out of bounds: %i' % plaintext)

        a, b = self.ciphertext, plaintext

        if len(a) == 2:
            prod_ciphertext = a[0]*b, a[1]*b
        else:
            prod_ciphertext = a*b
        return LabEncryptedNumber(self.mpk, prod_ciphertext)

    def _mul_encrypted(self, other):
        """Returns the Paillier encryption [[a*b-st]], given E(a) = (a-s,[[s]]) 
            and E(b) = (b-t,[[t]])

        Args:
          ciphertext (int,int): number by which to multiply the
            `LabEncryptedNumber`. 

        Returns:
          PaillierEncryptedNumber: return [[(a-s)(b-t)]] + Pai_mult([[t]],a-s) + 
            + Pai_mult([[s]],b-t)

        Raises:
          TypeError: if *self* or*other* is not a full ciphertext (int,int).

        """

        a, b = self.ciphertext, other.ciphertext

        if len(a) < 2:
            raise TypeError('Expected first factor to be a full LabHE encryption, not %s' %
                type(a))

        if len(b) < 2:
            raise TypeError('Expected second factor to be a full LabHE encryption, not %s' %
                type(b))

        prod_ciphertext = a[0]*b[0] + a[0]*b[1] + a[1]*b[0]

        return prod_ciphertext

    def mlt3(self, other1, other2, extra):
        """Returns the integer [[a * b * c - s * t * u]], given E(a) = (a-s,[[s]]), 
            E(b) = (b-t,[[t]]) and E(c) = (c-u,[[u]]), along with 
            [[s*t]], [[s*u]],[[t*u]]

        Args:
          self, other1,other2 are (int,[[int]]), extra is ([[int]],[[int]],[[int]])

        Returns:
          PaillierEncryptedNumber: return [[(a-s)(b-t)(c-u)]] + Pai_mult([[tu]],a-s) + 
            + Pai_mult([[su]],b-t) + Pai_mult([[st]],c-u) + Pai_mult([[u]],(a-s)*(b-t)) +
            Pai_mult([[t]],(a-s)*(c-u)) + Pai_mult([[s]],(b-t)*(c-u))

        Raises:
          TypeError: if *self* or*other* is not a full ciphertext (int,int).

        """
        if isinstance(other1, LabEncryptedNumber) and isinstance(other2, LabEncryptedNumber):
            if (isinstance(extra[0], paillier.EncryptedNumber) and isinstance(extra[1], paillier.EncryptedNumber) 
              and isinstance(extra[2], paillier.EncryptedNumber)):

                a, b, c = self.ciphertext, other1.ciphertext, other2.ciphertext
                s1s2, s1s3, s2s3 = extra[0], extra[1], extra[2]

                if len(a) < 2:
                    raise TypeError('Expected first factor to be a full LabHE encryption, not %s' %
                        type(a))

                if len(b) < 2:
                    raise TypeError('Expected second factor to be a full LabHE encryption, not %s' %
                        type(b))

                if len(c) < 2:
                    raise TypeError('Expected third factor to be a full LabHE encryption, not %s' %
                        type(b))

                prod_ciphertext = (a[0]*b[0]*c[0] + a[0]*s2s3 + b[0]*s1s3 + c[0]*s1s2 + 
                                    (a[0]*b[0])*c[1] + (a[0]*c[0])*b[1] + (b[0]*c[0])*a[1])

            else:
                raise TypeError('Need to have the extra information for multiplication.')
        else:
            raise TypeError('Need to have full LabHE encryptions')

        return prod_ciphertext

# To merge with mlt3
    def mlt4(self, other1, other2, other3, extra):
        """Returns the integer [[a * b * c * d - s * t * u * v]], given E(a) = (a-s,[[s]]), 
            E(b) = (b-t,[[t]]) and E(c) = (c-u,[[u]]), E(d-v,[[v]]) along with 
            [[s*t]], [[s*u]], [[s*v]], [[t*u]], [[t*v]], [[u*v]], [[s*t*u]], [[s*t*v]], 
            [[s*u*v]], [[t*u*v]]

        Args:
          self, other1,other2,other3 are (int,[[int]]), extra is ([[int]],[[int]],[[int]])

        Returns:
          PaillierEncryptedNumber: return [[a * b * c * d - s * t * u * v]]

        Raises:
          TypeError: if *self* or*other* is not a full ciphertext (int,int).

        """
        if (isinstance(other1, LabEncryptedNumber) and isinstance(other2, LabEncryptedNumber) and
             isinstance(other2, LabEncryptedNumber)):
            len_extra = len(extra)
            if len_extra == 10:
                flag = 1
                for k in range(len_extra):
                    flag = flag and isinstance(extra[k], paillier.EncryptedNumber)
                if (flag == 1):

                    a, b, c, d = self.ciphertext, other1.ciphertext, other2.ciphertext, other3.ciphertext
                    s1s2, s1s3, s1s4, s2s3, s2s4, s3s4  = extra[0], extra[1], extra[2], extra[3], extra[4], extra[5]
                    s1s2s3, s1s2s4, s1s3s4, s2s3s4 = extra[6], extra[7], extra[8], extra[9]

                    prod_ciphertext = (a[0]*b[0]*c[0]*d[0] + a[0]*s2s3s4 + b[0]*s1s3s4 + c[0]*s1s2s4 + 
                                        d[0]*s1s2s3 + (a[0]*b[0])*s3s4 + (a[0]*c[0])*s2s4 + (a[0]*d[0]*s2s3) +
                                        (b[0]*c[0])*s1s4 + (b[0]*d[0])*s1s3 + (c[0]*d[0])*s1s2 + 
                                        (a[0]*b[0]*c[0])*d[1] + (a[0]*b[0]*d[0])*c[1] + (a[0]*c[0]*d[0])*b[1] + 
                                        (b[0]*c[0]*d[0])*a[1])
                else:
                    raise TypeError('The extra information has to be Paillier encryptions.')

            else:
                raise TypeError('Need to have the extra information for multiplication.')
        else:
            raise TypeError('Need to have full LabHE encryptions')

        return prod_ciphertext